import multiprocessing
from functools import partial
import chemopore


def iterate(runner, n_iterations):
    print(runner.model)
    runner.iterate(n_iterations)


def pool_run(runners, n_iterations):
    iterate_partial = partial(iterate, n_iterations=n_iterations)
    pool = multiprocessing.Pool(processes=3)
    pool.map(iterate_partial, runners)
    pool.close()
    pool.join()


def pool_run_args(argses, output_every, n_iterations):
    runners = []
    for args in argses:
        output_dirname = chemopore.make_output_dirname(args)
        model = chemopore.AgentModel(**args)
        runner = chemopore.Runner(output_dirname, output_every, model,
                                  overwrite=True)
        runners.append(runner)
    pool_run(runners, n_iterations)
