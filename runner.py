import pickle
from os.path import join, basename, splitext, isdir
import os
import glob
import model


def f_to_i(f):
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=f_to_i)


def filename_to_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def make_output_dirname(args):
    fields = []
    for key, val in sorted(args.items()):
        if key == 'rc' and val is not None:
            val = len(args[key])
        fields.append('-'.join([key, model.format_parameter(val)]))
    return ','.join(fields)


class Runner(object):

    def __init__(self, output_dir, output_every, model=None):
        self.output_dir = output_dir
        self.output_every = output_every
        self.model = model

        if not isdir(self.output_dir):
            os.makedirs(self.output_dir)

        # If no model is provided, assume we are resuming from the output
        # directory and unpickle the most recent model from that.
        if self.model is None:
            output_filenames = get_filenames(self.output_dir)
            if output_filenames:
                self.model = filename_to_model(output_filenames[-1])
            else:
                raise IOError('Can not find any output pickles to resume from')

    def clear_dir(self):
        for snapshot in get_filenames(self.output_dir):
            assert snapshot.endswith('.pkl')
            os.remove(snapshot)

    def iterate(self, n=None, n_upto=None, t=None, t_upto=None):
        if t is not None:
            t_upto = self.model.t + t
        if t_upto is not None:
            n_upto = int(round(t_upto // self.model.dt))
        if n is not None:
            n_upto = self.model.i + n

        while self.model.i < n_upto:
            if not self.model.i % self.output_every:
                self.make_snapshot()
            self.model.iterate()

    def make_snapshot(self):
        filename = join(self.output_dir, '{:010d}.pkl'.format(self.model.i))
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def __repr__(self):
        info = '{}(out={}, model={})'
        return info.format(self.__class__.__name__, basename(self.output_dir),
                           self.model)
