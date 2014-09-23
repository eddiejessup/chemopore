import multiprocessing
from functools import partial
from runner import Runner, make_output_dirname, get_filenames
from model import AgentModel
from os.path import join


def iterate(runner, t_upto):
    print(runner.model)
    runner.iterate(t_upto=t_upto)


def pool_run(runners, t_upto):
    iterate_partial = partial(iterate, t_upto=t_upto)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.map_async(iterate_partial, runners).get(1e100)
    pool.close()
    pool.join()


def pool_run_args(argses, super_dirname, output_every, t_upto, resume):
    runners = []
    for args in argses:
        output_dirname = make_output_dirname(args)
        output_dirpath = join(super_dirname, output_dirname)
        if resume and get_filenames(output_dirpath):
            runner = Runner(output_dirpath, output_every)
        else:
            model = AgentModel(**args)
            runner = Runner(output_dirpath, output_every,
                            model=model)
            runner.clear_dir()
        runners.append(runner)
    pool_run(runners, t_upto)
