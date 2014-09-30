from runner import get_filenames, filename_to_model
import numpy as np
import dynamics
from ciabatta import pack
import multirun
from parameters import defaults, agent_defaults


def run_over_Dr(super_dirname, output_every, t_upto, resume,
                tumble, packing_fraction, D_rot_0s):
    args = defaults.copy()
    args.update(agent_defaults)
    args['tumble'] = tumble

    rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'],
                       pf=packing_fraction)
    args['rc'] = rc
    args['Rc'] = Rc

    argses = []
    for D_rot_0 in D_rot_0s:
        args['D_rot_0'] = D_rot_0
        argses.append(args.copy())
    multirun.pool_run_args(argses, super_dirname, output_every, t_upto, resume)


def run_over_phi(super_dirname, output_every, t_upto, resume,
                 tumble, phis):
    args = defaults.copy()
    args.update(agent_defaults)
    args['tumble'] = tumble

    argses = []
    for phi in phis:
        rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'], pf=phi)
        args['rc'] = rc
        args['Rc'] = Rc
        argses.append(args.copy())
    multirun.pool_run_args(argses, super_dirname, output_every, t_upto, resume)


def run_over_chi(super_dirname, output_every, t_upto, resume,
                 tumble, memory, chis):
    args = defaults.copy()
    args.update(agent_defaults)
    args['tumble'] = tumble
    args['memory'] = memory
    args['fixed_food_gradient'] = True

    argses = []
    for chi in chis:
        args['chi'] = chi
        argses.append(args.copy())
    multirun.pool_run_args(argses, super_dirname, output_every, t_upto, resume)


def measure_D_of_Dr(output_dirnames):
    D_rot_0s, Ds, Ds_err = [], [], []
    for output_dirname in output_dirnames:
        output_filenames = get_filenames(output_dirname)
        if not output_filenames:
            continue
        first_model = filename_to_model(output_filenames[0])
        recent_model = filename_to_model(output_filenames[-1])
        dr, dt = dynamics.model_to_dr_dt(recent_model, first_model)
        (D_mean, D_err, v_drift_mean, v_drift_err,
         D_total_mean, D_total_err) = dynamics.particle_dynamics(dr, dt)

        D_rot_0 = first_model.D_rot_0

        D_rot_0s.append(D_rot_0)
        Ds.append(D_total_mean)
        Ds_err.append(D_total_err)
    i_increasing_D_rot_0 = np.argsort(D_rot_0s)
    D_rot_0s = np.array(D_rot_0s)[i_increasing_D_rot_0]
    Ds = np.array(Ds)[i_increasing_D_rot_0]
    Ds_err = np.array(Ds_err)[i_increasing_D_rot_0]
    return D_rot_0s, Ds, Ds_err


def measure_D_of_phi(output_dirnames):
    phis, Ds, Ds_err = [], [], []
    for output_dirname in output_dirnames:
        output_filenames = get_filenames(output_dirname)
        if not output_filenames:
            continue
        first_model = filename_to_model(output_filenames[0])
        recent_model = filename_to_model(output_filenames[-1])
        dr, dt = dynamics.model_to_dr_dt(recent_model, first_model)
        (D_mean, D_err, v_drift_mean, v_drift_err,
         D_total_mean, D_total_err) = dynamics.particle_dynamics(dr, dt)

        phi = pack.n_to_pf(first_model.L[0], first_model.dim,
                           len(first_model.rc), first_model.Rc)
        phis.append(phi)
        Ds.append(D_total_mean)
        Ds_err.append(D_total_err)
    i_increasing_phi = np.argsort(phis)
    phis = np.array(phis)[i_increasing_phi]
    Ds = np.array(Ds)[i_increasing_phi]
    Ds_err = np.array(Ds_err)[i_increasing_phi]
    return phis, Ds, Ds_err


def measure_vd_of_chi(output_dirnames):
    chis, vds, vds_err = [], [], []
    for output_dirname in output_dirnames:
        output_filenames = get_filenames(output_dirname)
        if not output_filenames:
            continue
        first_model = filename_to_model(output_filenames[0])
        recent_model = filename_to_model(output_filenames[-1])
        dr, dt = dynamics.model_to_dr_dt(recent_model, first_model)
        (D_mean, D_err, v_drift_mean, v_drift_err,
         D_total_mean, D_total_err) = dynamics.particle_dynamics(dr, dt)

        chi = first_model.chi

        chis.append(chi)
        vds.append(v_drift_mean)
        vds_err.append(v_drift_err)
    i_increasing_chi = np.argsort(chis)
    chis = np.array(chis)[i_increasing_chi]
    vds = np.array(vds)[i_increasing_chi]
    vds_err = np.array(vds_err)[i_increasing_chi]
    return chis, vds, vds_err
