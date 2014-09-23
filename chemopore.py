import numpy as np
from ciabatta import utils
import pickle
from os.path import join, basename, splitext, isdir
import os
import glob
from ciabatta.distance import csep_periodic_close, cdist_sq_periodic
import fipy
from make_mesh import make_porous_mesh


def get_K(t, dt, tau):
    A = 0.5
    t_s = np.arange(0.0, t, dt)
    g_s = t_s / tau
    K = np.exp(-g_s) * (1.0 - A * (g_s + (g_s ** 2) / 2.0))
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    K /= np.sum(K * -t_s * dt)
    return K


def f_to_i(f):
    return int(splitext(basename(f))[0])


def get_filenames(dirname):
    filenames = glob.glob('{}/*.pkl'.format(dirname))
    return sorted(filenames, key=f_to_i)


def filename_to_model(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)


def format_parameter(p):
    if isinstance(p, float):
        return '{:.3g}'.format(p)
    elif p is None:
        return 'N'
    elif isinstance(p, bool):
        return '{:d}'.format(p)
    else:
        return '{}'.format(p)


def make_output_dirname(args):
    fields = []
    for key, val in sorted(args.items()):
        if key == 'rc':
            val = len(args[key])
        fields.append('-'.join([key, format_parameter(val)]))
    return ','.join(fields)


def make_and_run(output_dirname, output_every, model, overwrite, n_iterations):
    r = Runner(output_dirname, output_every, model=model, overwrite=True)
    r.iterate(n_iterations)


class Runner(object):
    def __init__(self, output_dir, output_every, model=None, overwrite=False):
        self.output_dir = output_dir
        self.output_every = output_every
        self.model = model
        self.overwrite = overwrite

        # If a model is provided, run that
        if self.model is not None:
            # If directory exists, clear it if we are overwriting, otherwise
            # raise an Exception.
            if isdir(self.output_dir):
                if self.overwrite:
                    for snapshot in get_filenames(self.output_dir):
                        assert snapshot.endswith('.pkl')
                        os.remove(snapshot)
                else:
                    raise IOError('Output directory exists: remove it or set '
                                  '`overwrite` flag to True')
            # If directory does not exist, create it.
            else:
                os.makedirs(self.output_dir)
        # If no model is provided, assume we are resuming from the output
        # directory and unpickle the most recent model from that.
        else:
            output_filenames = get_filenames(self.output_dir)
            if output_filenames:
                self.model = filename_to_model(output_filenames[-1])
            else:
                raise IOError('Can not find any output pickles to resume from')

    def iterate(self, n):
        for i in range(n):
            if not i % self.output_every:
                self.make_snapshot()
            self.model.iterate()

    def make_snapshot(self):
        filename = join(self.output_dir, '{:010d}.pkl'.format(self.model.i))
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)

    def __str__(self):
        info = '{}(out={}, model={})'
        return info.format(self.__class__.__name__, basename(self.output_dir),
                           self.model)


class Model(object):
    def __init__(self, L, dim, dt, rho_0, v_0, D_rot_0, chi, seed, rc, Rc, dx,
                 food_0, gamma, D_food):
        self.L = utils.pad_length(L, dim)
        self.dim = dim
        self.dt = dt
        self.rho_0 = rho_0
        self.v_0 = v_0
        self.D_rot_0 = D_rot_0
        self.chi = chi
        self.seed = seed
        self.rc = rc
        self.Rc = Rc
        self.dx = utils.pad_length(dx, dim)
        self.food_0 = food_0
        self.gamma = gamma
        self.D_food = D_food

        np.random.seed(self.seed)
        self.validate_parameters()

        if self.has_lattice():
            self.initialise_fields()
        self.initialise_equations()

        self.i, self.t = 0, 0.0

    def initialise_mesh(self):
        if self.has_obstacles():
            self.mesh = make_porous_mesh(self.rc, self.Rc, self.dx, self.L)
        elif self.dim == 1:
            self.mesh = fipy.Grid1D(Lx=self.L[0], dx=self.dx[0],
                                    origin=(-self.L[0] / 2.0,))
        elif self.dim == 2:
            self.mesh = fipy.Grid2D(Lx=self.L[0], Ly=self.L[1],
                                    dx=self.dx[0], dy=self.dx[1],
                                    origin=((-self.L[0] / 2.0,),
                                            (-self.L[1] / 2.0,)))

    def validate_parameters(self):
        if self.dim == 1 and self.has_obstacles():
            raise Exception('Cannot have obstacles in 1D.')
        if self.v_0 and self.Rc and self.Rc / (self.v_0 * self.dt) < 10.0:
            raise Exception('Time-step too large: particle crosses obstacles '
                            'too fast.')
        if self.D_rot_0 and np.pi / np.sqrt(self.D_rot_0 * self.dt) < 50.0:
            raise Exception('Time-step too large: particle randomises '
                            'direction too fast.')

    def has_obstacles(self):
        return self.rc is not None and len(self.rc) and self.Rc

    def has_food_field(self):
        return self.has_lattice() and (self.food_0 is not None or
                                       self.gamma is not None or
                                       self.D_food is not None)

    def has_lattice(self):
        return any([e is not None for e in self.dx])

    def initialise_density_field(self):
        self.rho = fipy.CellVariable(name="density", mesh=self.mesh,
                                     hasOld=True)

    def initialise_polarisation_field(self):
        self.p = fipy.CellVariable(name="polarisation", mesh=self.mesh,
                                   value=0.0, rank=1, hasOld=True)

    def initialise_food_field(self):
        self.food = fipy.CellVariable(name="food", mesh=self.mesh,
                                      value=self.food_0, hasOld=True)

    def initialise_fields(self):
        self.initialise_mesh()
        self.initialise_density_field()
        self.initialise_polarisation_field()
        if self.has_food_field():
            self.initialise_food_field()

    def initialise_food_equation(self):
        self.food_PDE = (fipy.TransientTerm() ==
                         fipy.DiffusionTerm(coeff=self.D_food) -
                         fipy.ImplicitSourceTerm(coeff=self.gamma * self.rho))

    def initialise_equations(self):
        if self.has_food_field():
            self.initialise_food_equation()

    def iterate(self):
        self.update_fields()
        self.i += 1
        self.t += self.dt

    def __str__(self):
        f = format_parameter
        info = ('{}(d={}, L={}, Rc={}, nc={}, chi={}, D_rot_0={})')
        return info.format(self.__class__.__name__, self.dim, self.L,
                           f(self.Rc), len(self.rc), f(self.chi),
                           f(self.D_rot_0))


class AgentModel(Model):
    def __init__(self, L, dim, dt, rho_0, v_0, D_rot_0, chi, seed, rc, Rc, dx,
                 food_0, gamma, D_food,
                 tumble, dt_chemo, memory, t_mem):
        self.tumble = tumble
        self.dt_chemo = dt_chemo
        self.memory = memory
        self.t_mem = t_mem
        Model.__init__(self, L, dim, dt, rho_0, v_0, D_rot_0, chi, seed, rc,
                       Rc, dx, food_0, gamma, D_food)

        self.calculate_n()
        self.initialise_particles()
        if self.chi:
            self.initialise_chemotaxis()

    def calculate_n(self):
        V = np.product(self.L)
        self.n = int(round(self.rho_0 * V))
        self.rho_0 = self.n / V

    def validate_parameters(self):
        Model.validate_parameters(self)
        if self.dim == 1 and not self.tumble:
            raise Exception('Cannot have rotational diffusion in 1D,'
                            'particles must do tumbling.')
        if self.chi and self.dt_chemo < self.dt:
            raise Exception('Chemotaxis time-step must be at least '
                            'the system timestep.')

    def initialise_particles(self):
        # Intitialise velocities
        self.v = self.v_0 * utils.sphere_pick(n=self.n, d=self.dim)

        # Initialise positions
        self.r = np.empty_like(self.v)
        for i in range(self.n):
            while True:
                for i_dim in range(self.dim):
                    self.r[i, i_dim] = np.random.uniform(-self.L[i_dim] / 2.0,
                                                         self.L[i_dim] / 2.0)
                if self.has_obstacles():
                    sep_sq = csep_periodic_close(self.r[np.newaxis, i],
                                                 self.rc, self.L)[1][0]
                    if sep_sq > self.Rc ** 2:
                        break
                else:
                    break

        self.wraps = np.zeros_like(self.r, dtype=np.int)

        self.D_rot = self.D_rot_0

    def initialise_chemotaxis(self):
        # Calculate best dt_chemo that can be managed
        # given that it must be an integer multiple of dt.
        # Update chemotaxis every so many iterations
        self.every_chemo = int(round(self.dt_chemo / self.dt))
        # derive effective dt_chemo from this
        self.dt_chemo = self.every_chemo * self.dt

        if self.memory:
            t_rot_0 = 1.0 / self.D_rot_0
            self.K_dt_chemo = get_K(self.t_mem, self.dt_chemo,
                                    t_rot_0) * self.dt_chemo
            self.c_mem = np.zeros([self.n, len(self.K_dt_chemo)])

    def get_p(self):
        self.update_p()
        return self.p

    def get_density(self):
        self.update_density()
        return self.rho

    def get_inds_close(self):
        return np.argmin(cdist_sq_periodic(self.r,
                                           self.mesh.cellCenters.value.T,
                                           self.L), axis=1)

    def get_unwrapped_r(self):
        return self.r + self.wraps * self.L

    def update_D_rot(self):
        if self.chi:
            # Update D_rot every `ever_chemo` iterations
            if not self.i % self.every_chemo:
                inds_close = self.get_inds_close()
                if self.memory:
                    c_cur = self.food[inds_close]
                    self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
                    self.c_mem[:, 0] = c_cur
                    v_dot_grad_c = np.sum(self.c_mem * self.K_dt_chemo, axis=1)
                else:
                    grad_c = self.food.grad().T[inds_close]
                    v_dot_grad_c = np.sum(self.v * grad_c, axis=-1)

                # Calculate fitness and chemotactic rotational diffusion
                # constant
                f = self.chi * v_dot_grad_c / self.v_0
                self.D_rot = self.D_rot_0 * (1.0 - f)
                # Check rotational diffusion constant is physically meaningful
                if np.any(self.D_rot <= 0.0) and (not self.memory or
                                                  self.t > self.t_mem):
                    raise Exception

    def update_noise(self):
        # Do rotational diffusion / tumbling
        if self.tumble:
            tumblers = np.random.uniform(size=self.n) < self.D_rot * self.dt
            self.v[tumblers] = self.v_0 * utils.sphere_pick(n=np.sum(tumblers),
                                                            d=self.dim)
        else:
            self.v = utils.rot_diff(self.v, self.D_rot, self.dt)

    def update_positions(self):
        # Keep a copy of old positions in case a particle needs to revert
        # due to a collision.
        r_old = self.r.copy()

        # Make particles swim.
        self.r += self.v * self.dt

        # Wrap particles if they go outside the system boundaries.
        wraps_cur = np.zeros_like(self.wraps, dtype=np.int)
        for i_dim in range(self.dim):
            wraps_cur[:, i_dim] += self.r[:, i_dim] > self.L[i_dim] / 2.0
            wraps_cur[:, i_dim] -= self.r[:, i_dim] < -self.L[i_dim] / 2.0
        self.wraps += wraps_cur
        self.r -= wraps_cur * self.L

        # If there are any obstacles, with a number above zero, and a radius
        # above zero.
        if self.has_obstacles():
            # Get separation vectors to each closest sphere.
            sep, sep_sq = csep_periodic_close(self.r, self.rc, self.L)

            # Get distance to each closest sphere.
            sep_mag = np.sqrt(sep_sq)
            # Those that collide have a separation distance < obstacles radius.
            colls = sep_mag < self.Rc

            # Those that collide go back to there old position.
            self.r[colls] = r_old[colls]
            # Must re-update wrap because they didn't actually move.
            self.wraps[colls] -= wraps_cur[colls]
            # Must find separation vectors again for collided particles
            # because they've moved back
            sep_colls, sep_sq_colls = csep_periodic_close(self.r[colls],
                                                          self.rc, self.L)
            sep_mag_colls = np.sqrt(sep_sq_colls)
            # Find the unit vector between collided particles and their
            # nearest sphere.
            u_sep = sep_colls / sep_mag_colls[:, np.newaxis]
            # Find the particle's velocity component perpendicular to the
            # sphere surface.
            # Its magnitude is the dot product of the particle's velocity
            # with the unit separation vector to the sphere.
            v_dot_u_sep = np.sum(self.v[colls] * u_sep, axis=-1)
            # And its direction is in the direction of the separation vector.
            v_perp = v_dot_u_sep[:, np.newaxis] * u_sep
            # Reflect the velocity for collided particles in the plane
            # tangential to the obstacle surface.
            self.v[colls] -= 2.0 * v_perp

    def update_p(self):
        n = np.zeros(self.rho.shape, dtype=np.int)
        self.p.setValue(0.0)
        for i, ind_close in enumerate(self.get_inds_close()):
            n[ind_close] += 1
            self.p[:, ind_close] += self.v[i]
        self.p[:, n > 0] = self.p[:, n > 0].value / n[np.newaxis, n > 0]
        self.p.setValue(self.p / self.v_0)

    def update_density(self):
        n = np.zeros(self.rho.shape, dtype=np.int)
        for ind_close in self.get_inds_close():
            n[ind_close] += 1
        self.rho.setValue(n / self.mesh.cellVolumes)

    def update_fields(self):
        if self.has_food_field():
            self.update_density()
            self.food_PDE.solve(var=self.food, dt=self.dt)

    def iterate(self):
        self.update_D_rot()
        self.update_noise()
        self.update_positions()
        Model.iterate(self)

    def __str__(self):
        f = format_parameter
        info = ('{}(d={}, L={}, Rc={}, nc={}, chi={}, D_rot_0={}, '
                'tumble={}, n={}, memory={})')
        return info.format(self.__class__.__name__, self.dim, self.L,
                           f(self.Rc), len(self.rc), f(self.chi),
                           f(self.D_rot_0),
                           f(self.tumble), self.n, f(self.memory))


class CoarseModel(Model):
    def initialise_D_rot_field(self):
        self.D_rot = fipy.CellVariable(name="rotational diffusion constant",
                                       rank=1, mesh=self.mesh,
                                       value=self.D_rot_0)

    def initialise_fields(self):
        Model.initialise_fields(self)
        self.initialise_D_rot_field()

        # m_half = self.L[0] // self.dx[0] // 2
        self.rho[...] = np.random.normal(loc=self.rho_0,
                                         scale=np.sqrt(self.rho_0),
                                         size=self.rho.shape)
        # scale = 0.5
        # self.rho[...] = np.exp(-np.square(self.mesh.cellCenters[0] /
        #                        scale) / 2.0)
        self.rho[...] /= self.rho.sum()
        V = np.product(self.L)
        dV = np.product(self.dx)
        self.rho[...] *= V * self.rho_0 / dV

    def initialise_density_equation(self):
        self.density_PDE = (fipy.TransientTerm() == -self.v_0 *
                            fipy.PowerLawConvectionTerm(coeff=self.p))

    def initialise_orientation_equation(self):
        self.orientation_PDE = (fipy.TransientTerm() ==
                                fipy.ImplicitSourceTerm(coeff=-self.D_rot) -
                                (self.v_0 / 2.0) * self.rho.grad / self.rho)

    def initialise_equations(self):
        Model.initialise_equations(self)
        self.initialise_density_equation()
        self.initialise_orientation_equation()

    def get_p(self):
        return self.p

    def update_D_rot(self):
        if self.chi:
            u_p_dot_grad_c = self.p.dot(self.food.grad) / self.p.mag

            # Calculate fitness and chemotactic rotational diffusion constant
            f = np.where(np.isfinite(u_p_dot_grad_c),
                         self.chi * u_p_dot_grad_c, 0.0)
            self.D_rot.setValue(self.D_rot_0 * (1.0 - f))
            # Check rotational diffusion constant is physically meaningful
            if np.any(self.D_rot <= 0.0):
                raise Exception

    def update_fields(self):
        self.rho.updateOld()
        self.p.updateOld()
        if self.has_food_field():
            self.food.updateOld()
        self.density_PDE.solve(var=self.rho, dt=self.dt)
        self.orientation_PDE.solve(var=self.p, dt=self.dt)
        if self.has_food_field():
            self.food_PDE.solve(var=self.food, dt=self.dt)

    def iterate(self):
        self.update_D_rot()
        Model.iterate(self)
