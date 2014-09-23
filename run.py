from chemopore import get_filenames, filename_to_model
import numpy as np
import dynamics
from ciabatta import pack
import multirun
from parameters import defaults, agent_defaults


def run_D_of_Dr(super_dirname, output_every, n_iterations,
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
    multirun.pool_run_args(argses, super_dirname, output_every, n_iterations)


def run_D_of_phi(super_dirname, output_every, n_iterations, tumble, phis):
    args = defaults.copy()
    args.update(agent_defaults)
    args['tumble'] = tumble

    argses = []
    for phi in phis:
        rc, Rc = pack.pack(args['dim'], args['Rc'], args['seed'], pf=phi)
        args['rc'] = rc
        args['Rc'] = Rc
        argses.append(args)
    multirun.pool_run_args(argses, super_dirname, output_every, n_iterations)


def measure_D_of_Dr(output_dirnames):
    D_rot_0s, Ds, Ds_err = [], [], []
    for output_dirname in output_dirnames:
        first_model = filename_to_model(get_filenames(output_dirname)[0])
        D_rot_0 = first_model.D_rot_0
        D_of_t, D_of_t_err = dynamics.particle_dynamics(output_dirname)[-2:]
        D, D_err = D_of_t[-1], D_of_t_err[-1]

        D_rot_0s.append(D_rot_0)
        Ds.append(D)
        Ds_err.append(D_err)
    return D_rot_0s, Ds, Ds_err


def measure_D_of_phi(output_dirnames):
    phis, Ds, Ds_err = [], [], []
    for output_dirname in output_dirnames:
        first_model = filename_to_model(get_filenames(output_dirname)[0])
        phi = pack.n_to_pf(first_model.L[0], first_model.dim,
                           len(first_model.rc), first_model.Rc)
        D_of_t, D_of_t_err = dynamics.particle_dynamics(output_dirname)[-2:]
        D, D_err = D_of_t[-1], D_of_t_err[-1]

        phis.append(phi)
        Ds.append(D)
        Ds_err.append(D_err)
    return phis, Ds, Ds_err
