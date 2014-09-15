import numpy as np
from ciabatta import utils, pack
import pickle
from os.path import join, basename, splitext
import glob
from ciabatta.distance import csep_periodic_close, cdist_sq_periodic
import fipy


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


class Runner(object):
    def __init__(self, output_dir, output_every, model=None, input_dir=None):
        if model is not None:
            self.model = model
        elif input_dir is not None:
            recent_filename = get_filenames(input_dir)[-1]
            self.model = filename_to_model(recent_filename)

        self.output_dir = output_dir
        self.output_every = output_every
        utils.makedirs_safe(self.output_dir)

    def iterate(self, n):
        for i in range(n):
            if not i % self.output_every:
                self.make_snapshot()
            self.model.iterate()

    def make_snapshot(self):
        filename = join(self.output_dir, '{:010d}.pkl'.format(self.model.i))
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)


def pad_length(x, dim):
    try:
        x[0]
    except TypeError:
        x = dim * [x]
    return np.array(x)


class Model(object):
    def __init__(self,
                 L, dim, dt,
                 n, v_0, D_rot_0, tumble,
                 chi, dt_chemo, memory, t_mem,
                 seed,
                 rc, Rc,
                 dx, food_0, gamma, D_food):
        self.L = pad_length(L, dim)
        self.dim = dim
        self.dt = dt
        self.n = n
        self.v_0 = v_0
        self.D_rot_0 = D_rot_0
        self.tumble = tumble
        self.chi = chi
        self.dt_chemo = dt_chemo
        self.memory = memory
        self.t_mem = t_mem
        self.seed = seed
        self.rc = rc
        self.Rc = Rc
        self.dx = pad_length(dx, dim)
        self.food_0 = food_0
        self.gamma = gamma
        self.D_food = D_food

        np.random.seed(self.seed)
        self.validate_parameters()
        self.initialise_particles()
        if self.chi:
            self.initialise_chemotaxis()
        self.initialise_fields()
        self.i, self.t = 0, 0.0

    def validate_parameters(self):
        if self.v_0 and min(self.L) / (self.v_0 * self.dt) < 1000.0:
            raise Exception('Time-step too large: particle crosses system '
                            'too fast')
        if self.v_0 and self.Rc and self.Rc / (self.v_0 * self.dt) < 10.0:
            raise Exception('Time-step too large: particle crosses obstacles '
                            'too fast')
        if self.D_rot_0 and np.pi / np.sqrt(self.D_rot_0 * self.dt) < 50.0:
            raise Exception('Time-step too large: particle randomises '
                            'direction too fast')

    def has_obstacles(self):
        return self.rc is not None and len(self.rc) and self.Rc

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

    def initialise_chemotaxis(self):
        if self.dt_chemo < self.dt:
            raise Exception('Chemotaxis time-step must be at least '
                            'the system timestep')
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
        else:
            self.grad_c = np.array([1.0] + (self.dim - 1) * [0.0])

    def initialise_fields(self):
        self.mesh = fipy.Grid2D(Lx=self.L[0], Ly=self.L[1],
                                dx=self.dx[0], dy=self.dx[1])

        self.r_mesh = self.mesh.cellCenters.value.T - self.L[np.newaxis] / 2.0

        # Set up density field
        self.rho = fipy.CellVariable(name="density", mesh=self.mesh, value=0.0)

        # Set up food field
        self.food = fipy.CellVariable(name="food", mesh=self.mesh,
                                      value=self.food_0)

        self.food_PDE = (fipy.TransientTerm() ==
                         fipy.DiffusionTerm(coeff=self.D_food) -
                         fipy.ImplicitSourceTerm(coeff=self.gamma * self.rho))

    def get_inds_close(self):
        return np.argmin(cdist_sq_periodic(self.r, self.r_mesh, self.L),
                         axis=1)

    def update_D_rot(self):
        if self.chi:
            # Update D_rot every `ever_chemo` iterations
            if not self.i % self.every_chemo:
                if self.memory:
                    c_cur = (self.r + self.wraps * self.L)[:, 0]
                    self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
                    self.c_mem[:, 0] = c_cur
                    v_dot_grad_c = np.sum(self.c_mem * self.K_dt_chemo, axis=1)
                else:
                    v_dot_grad_c = np.sum(self.v * self.grad_c, axis=-1)

                # Calculate fitness and chemotactic rotational diffusion
                # constant
                f = self.chi * v_dot_grad_c / self.v_0
                self.D_rot = self.D_rot_0 * (1.0 - f)
                # Check rotational diffusion constant is physically meaningful
                if np.any(self.D_rot <= 0.0) and (not self.memory or
                                                  self.t > self.t_mem):
                    raise Exception
        else:
            self.D_rot = self.D_rot_0

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

    def update_density(self):
        self.rho.value[...] = 0.0
        inds_close = self.get_inds_close()
        for ind_close in inds_close:
            self.rho.value[ind_close] += 1.0 / self.mesh.cellVolumes[ind_close]
        self.rho.setValue(self.rho.value)

    def update_fields(self):
        self.update_density()
        self.food_PDE.solve(var=self.food, dt=self.dt)

    def iterate(self):
        self.update_D_rot()
        self.update_noise()
        self.update_positions()
        self.update_fields()

        self.i += 1
        self.t += self.dt


def save_medium(dirname, r, R, L, seed):
    n, dim = r.shape
    pf = pack.n_to_pf(L, dim, n, R)
    filename = 'medium_d_{}_n_{}_pf_{}_R_{}_s_{}'.format(dim, n, pf, R, seed)
    path = join(dirname, filename)
    np.savez(path, r=r, R=R, L=L, seed=seed)
