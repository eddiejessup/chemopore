from runner import get_filenames, filename_to_model
import numpy as np
import dynamics
from ciabatta import pack
import multirun
from parameters import defaults, agent_defaults


def run_param_sweep(super_dirname, output_every, t_upto, resume,
                    tumbles, memorys,
                    chis, phis, D_rot_0s):
    args = defaults.copy()
    args.update(agent_defaults)
    args['fixed_food_gradient'] = True

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

    argses = []
    for tumble in tumbles:
        args['tumble'] = tumble
        for memory in memorys:
            args['memory'] = memory
            for phi in phis:
                rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'],
                                   pf=phi)
                args['rc'] = rc
                args['Rc'] = Rc
                for chi in chis:
                    args['chi'] = chi
                    for D_rot_0 in D_rot_0s:
                        args['D_rot_0'] = D_rot_0
                        argses.append(args.copy())
    multirun.pool_run_args(argses, super_dirname, output_every, t_upto, resume)


def measure_y_of_x(output_dirnames, x_key, y_key):
    xs, ys, ys_err = [], [], []
    for output_dirname in output_dirnames:
        output_filenames = get_filenames(output_dirname)
        if not output_filenames:
            continue
        first_model = filename_to_model(output_filenames[0])
        recent_model = filename_to_model(output_filenames[-1])
        dr, dt = dynamics.model_to_dr_dt(recent_model, first_model)
        (D_mean, D_err, v_drift_mean, v_drift_err,
         D_total_mean, D_total_err) = dynamics.particle_dynamics(dr, dt)

        if x_key == 'chi':
            x = first_model.chi
        elif x_key == 'Dr':
            x = first_model.D_rot_0
        elif x_key == 'phi':
            x = pack.n_to_pf(first_model.L[0], first_model.dim,
                             len(first_model.rc), first_model.Rc)
        else:
            raise Exception('Unknown independent variable')

        if y_key == 'D':
            y = D_total_mean
            y_err = D_total_err
        elif y_key == 'vd':
            y = v_drift_mean[0]
            y_err = v_drift_err[0]
        else:
            raise Exception('Unknown dependent variable')

        xs.append(x)
        ys.append(y)
        ys_err.append(y_err)

    i_increasing_x = np.argsort(xs)
    xs = np.array(xs)[i_increasing_x]
    ys = np.array(ys)[i_increasing_x]
    ys_err = np.array(ys_err)[i_increasing_x]
    return xs, ys, ys_err
