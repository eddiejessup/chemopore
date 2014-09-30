from runner import get_filenames, filename_to_model
import numpy as np
import dynamics
from ciabatta import pack
import multirun
from parameters import defaults, agent_defaults


def run_over_x(super_dirname, output_every, t_upto, resume,
               tumble, memory,
               x_key, chi, phi, D_rot_0):
    args = defaults.copy()
    args.update(agent_defaults)
    args['tumble'] = tumble
    args['fixed_food_gradient'] = True
    args['memory'] = memory

    if x_key == 'chi':
        xs = chi
    else:
        args['chi'] = chi
    if x_key == 'D_rot_0':
        xs = D_rot_0
    else:
        args['D_rot_0'] = D_rot_0
    if x_key == 'phi':
        xs = phi
    else:
        rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'],
                           pf=phi)
        args['rc'] = rc
        args['Rc'] = Rc

    argses = []
    for x in xs:
        if x_key == 'chi':
            args['chi'] = x
        elif x_key == 'phi':
            rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'], pf=x)
            args['rc'] = rc
            args['Rc'] = Rc
        elif x_key == 'D_rot_0':
            args['D_rot_0'] = x
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
