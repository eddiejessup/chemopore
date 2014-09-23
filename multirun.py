import multiprocessing
from functools import partial
import chemopore
from os.path import join


def iterate(runner, n_iterations):
    print(runner.model)
    runner.iterate(n_iterations)


def pool_run(runners, n_iterations):
    iterate_partial = partial(iterate, n_iterations=n_iterations)
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
    pool.map_async(iterate_partial, runners).get(1e100)
    pool.close()
    pool.join()


def pool_run_args(argses, super_dirname, output_every, n_iterations, resume):
    runners = []
    for args in argses:
        output_dirname = chemopore.make_output_dirname(args)
        output_dirpath = join(super_dirname, output_dirname)
        if resume:
            model = None
            overwrite = False
        else:
            model = chemopore.AgentModel(**args)
            overwrite = True
        runner = chemopore.Runner(output_dirpath, output_every, model,
                                  overwrite)
        runners.append(runner)
    pool_run(runners, n_iterations)
