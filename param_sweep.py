from runner import get_filenames, filename_to_model
import numpy as np
import dynamics
from ciabatta import pack
import multirun
from parameters import defaults, agent_defaults
from itertools import product
from scipy.stats import sem


def run_param_sweep(super_dirname, output_every, t_upto, resume,
                    tumbles, memorys,
                    chis, phis, D_rot_0s,
                    seeds,
                    **kwargs):
    args = defaults.copy()
    args.update(agent_defaults)
    args['fixed_food_gradient'] = True
    args.update(kwargs)

    if isinstance(chis, float):
        chis = [chis]
    if isinstance(phis, float):
        phis = [phis]
    if isinstance(D_rot_0s, float):
        D_rot_0s = [D_rot_0s]
    if isinstance(tumbles, bool):
        tumbles = [tumbles]
    if isinstance(memorys, bool):
        memorys = [memorys]
    if isinstance(seeds, int):
        seeds = [seeds]

    argses = []
    for seed in seeds:
        args['seed'] = seed
        for phi in phis:
            rc, Rc = pack.pack(args['dim'], args['Rc'], args['L'],
                               seed=args['seed'], pf=phi)
            args['rc'] = rc
            args['Rc'] = Rc
            for tumble, memory, chi, D_rot_0 in product(tumbles, memorys,
                                                        chis, D_rot_0s):
                args['tumble'] = tumble
                args['memory'] = memory
                args['chi'] = chi
                args['D_rot_0'] = D_rot_0
                argses.append(args.copy())
    multirun.pool_run_args(argses, super_dirname, output_every, t_upto, resume)


def group_seeds(output_dirnames, x_key):
    seed_groups = {}
    for output_dirname in output_dirnames:
        output_filenames = get_filenames(output_dirname)
        if not output_filenames:
            continue
        model = filename_to_model(output_filenames[0])

        if x_key == 'chi':
            x = model.chi
        elif x_key == 'Dr':
            x = model.D_rot_0
        elif x_key == 'phi':
            x = pack.n_to_pf(model.L[0], model.dim, len(model.rc), model.Rc)
        else:
            raise Exception('Unknown independent variable')

        if x in seed_groups:
            seed_groups[x].append(output_dirname)
        else:
            seed_groups[x] = [output_dirname]
    return seed_groups


def output_dirname_to_y(output_dirname, y_key):
    output_filenames = get_filenames(output_dirname)
    first_model = filename_to_model(output_filenames[0])
    recent_model = filename_to_model(output_filenames[-1])
    dr, dt = dynamics.model_to_dr_dt(recent_model, first_model)
    (D_mean, D_err, v_drift_mean, v_drift_err,
     D_total_mean, D_total_err) = dynamics.particle_dynamics(dr, dt)
    if y_key == 'D':
        return D_total_mean, D_total_err
    elif y_key == 'vd':
        return v_drift_mean[0], v_drift_err[0]
    else:
        raise Exception('Unknown dependent variable')


def measure_y_of_x(output_dirnames, x_key, y_key):
    xs, ys, ys_err = [], [], []
    seed_groups = group_seeds(output_dirnames, x_key)
    for x, output_dirnames in seed_groups.items():
        ys_micro, ys_micro_err = [], []
        for output_dirname in output_dirnames:
            try:
                y_micro, y_micro_err = output_dirname_to_y(output_dirname,
                                                           y_key)
            except IndexError:
                continue
            ys_micro.append(y_micro)
            ys_micro_err.append(y_micro_err)

        if len(ys_micro) > 1:
            ys_err.append(sem(ys_micro))
        else:
            ys_err.append(ys_micro_err[0])
        ys.append(np.mean(ys_micro))
        xs.append(x)

    i_increasing_x = np.argsort(xs)
    xs = np.array(xs)[i_increasing_x]
    ys = np.array(ys)[i_increasing_x]
    ys_err = np.array(ys_err)[i_increasing_x]
    return xs, ys, ys_err
