from __future__ import print_function
import matplotlib.pyplot as plt
from runner import get_filenames, filename_to_model
from ciabatta import pack, ejm_rcparams
from fipy import MatplotlibViewer, MatplotlibVectorViewer
import numpy as np

ejm_rcparams.set_pretty_plots(False, False)


def plot_coarse(dirname):
    filenames = get_filenames(dirname)

    model_0 = filename_to_model(filenames[0])

    plt.show()
    plt.ion()

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs = axs.flatten()

    ax_rho, ax_p, ax_food, ax_food_grad = axs

    for ax in axs:
        ax.set_xlim(-model_0.L[0] / 2.0, model_0.L[0] / 2.0)
        ax.set_ylim(-model_0.L[1] / 2.0, model_0.L[1] / 2.0)

    view_args = {'xmin': -model_0.L[0] / 2.0, 'xmax': model_0.L[0] / 2.0,
                 'ymin': -model_0.L[1] / 2.0, 'ymax': model_0.L[1] / 2.0,
                 'cmap': ejm_rcparams.reds}

    rho_viewer = MatplotlibViewer(vars=model_0.rho,
                                  axes=ax_rho,
                                  colorbar=None,
                                  **view_args)

    p_viewer = MatplotlibViewer(vars=model_0.get_p(),
                                axes=ax_p, colorbar=None,
                                **view_args)

    food_viewer = MatplotlibViewer(vars=model_0.food,
                                   axes=ax_food,
                                   colorbar=None,
                                   **view_args)

    food_grad_viewer = MatplotlibVectorViewer(vars=model_0.food.grad,
                                              axes=ax_food_grad,
                                              fig_aspect='auto',
                                              scale=2,
                                              **view_args)

    # fig.set_tight_layout(True)

    viewers = [rho_viewer, p_viewer, food_viewer, food_grad_viewer]
    for viewer in viewers:
        viewer.plotMesh()

    for filename in filenames:
        model = filename_to_model(filename)

        for viewer, var in zip(viewers, [model.rho, model.get_p(),
                                         model.food, model.food.grad]):
            viewer.vars[0] = var
            viewer.plot()

        fig.canvas.draw()
        raw_input()


def plot(dirname):
    filenames = get_filenames(dirname)

    model_0 = filename_to_model(filenames[0])
    L = model_0.L

    plt.show()
    plt.ion()

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    axs = axs.flatten()

    ax_particles, ax_rho, ax_food, ax_food_grad = axs

    for ax in axs:
        ax.set_xlim(-L[0] / 2.0, L[0] / 2.0)
        ax.set_ylim(-L[1] / 2.0, L[1] / 2.0)

    particle_position_plot = ax_particles.quiver([], [], scale=10.0)

    food_viewer = MatplotlibViewer(vars=model_0.food,
                                   xmin=-L[0] / 2.0, xmax=L[0] / 2.0,
                                   ymin=-L[1] / 2.0, ymax=L[1] / 2.0,
                                   axes=ax_food,
                                   colorbar=None,
                                   cmap=ejm_rcparams.reds)

    rho_viewer = MatplotlibViewer(vars=model_0.rho,
                                  xmin=-L[0] / 2.0, xmax=L[0] / 2.0,
                                  ymin=-L[1] / 2.0, ymax=L[1] / 2.0,
                                  axes=ax_rho,
                                  colorbar=None,
                                  cmap=ejm_rcparams.reds)

    food_grad_viewer = MatplotlibVectorViewer(vars=model_0.food.grad,
                                              xmin=-L[0] / 2.0,
                                              xmax=L[0] / 2.0,
                                              ymin=-L[1] / 2.0,
                                              ymax=L[1] / 2.0,
                                              axes=ax_food_grad,
                                              fig_aspect='auto',
                                              scale=2)

    if model_0.has_obstacles():
        pack.draw_medium(model_0.rc, model_0.Rc, L, 1, ax_particles)

    # fig.set_tight_layout(True)

    for view in [rho_viewer, food_viewer]:
        view.plotMesh()

    for filename in filenames:
        model = filename_to_model(filename)

        particle_position_plot.set_offsets(model.r)
        particle_position_plot.set_UVC(model.v[:, 0], model.v[:, 1])

        food_viewer.vars[0] = model.food
        rho_viewer.vars[0] = model.rho
        food_grad_viewer.vars[0] = model.food.grad

        food_viewer.plot()
        rho_viewer.plot()
        food_grad_viewer.plot()
        fig.canvas.draw()
        raw_input()

        # v_drift = (model.get_unwrapped_r()[:, 0] - model_0.r[:, 0]) / model.t
        # print(model.t, np.mean(v_drift), np.std(v_drift))


def plot_1d(dirname):
    filenames = get_filenames(dirname)

    model_0 = filename_to_model(filenames[0])
    L = model_0.L

    # plt.show()
    # plt.ion()

    fig, axs = plt.subplots(4, 1, sharex=True)

    ax_rho, ax_p, ax_food, ax_Dr = axs

    ax_rho.set_title('Density')
    ax_p.set_title('Polarisation')
    ax_food.set_title('Food concentration')
    ax_Dr.set_title('Rotational diffusion')
    for ax in axs:
        ax.set_xlim(-L[0] / 2.0, L[0] / 2.0)

    xs = model_0.rho.mesh.cellCenters.value[0]
    i_sort = np.argsort(xs)
    xs = xs[i_sort]

    rho_plot = ax_rho.plot(xs, xs)[0]
    p_plot = ax_p.plot(xs, xs)[0]
    food_plot = ax_food.plot(xs, xs)[0]
    Dr_plot = ax_Dr.plot(xs, xs)[0]

    first = True
    for filename in filenames:
        print(filename)
        model = filename_to_model(filename)

        rhos = model.rho[i_sort]
        rho_plot.set_ydata(rhos)
        ps = model.get_p()[0, i_sort].grad[0]
        p_plot.set_ydata(ps)
        foods = model.food[i_sort]
        food_plot.set_ydata(foods)
        Drs = model.D_rot[0, i_sort]
        Dr_plot.set_ydata(Drs)

        if first:
            ax_rho.set_ylim(0.0, 1.2 * rhos.max())
            # ax_rho.set_ylim(0.0, 5.0)
            ps_lim = 1.2 * np.abs(ps).max()
            ax_p.set_ylim(-ps_lim, ps_lim)
            ax_p.set_ylim(-5.0, 5.0)
            ax_food.set_ylim(0.0, 1.2 * foods.max())
            ax_Dr.set_ylim(0.0, 2.1 * Drs.max())
            first = False

        fig.canvas.draw()
        raw_input()


def plot_agent_nofield(dirname):
    filenames = get_filenames(dirname)

    model_0 = filename_to_model(filenames[0])
    L = model_0.L

    plt.show()
    plt.ion()

    fig, axs = plt.subplots(1, 1)

    ax_particles = axs

    ax_particles.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_particles.set_ylim(-L[1] / 2.0, L[1] / 2.0)

    particle_position_plot = ax_particles.quiver([], [], scale=100.0)

    if model_0.has_obstacles():
        pack.draw_medium(model_0.rc, model_0.Rc, L, 1, ax_particles)

    # fig.set_tight_layout(True)

    for filename in filenames:
        model = filename_to_model(filename)

        particle_position_plot.set_offsets(model.r)
        particle_position_plot.set_UVC(model.v[:, 0], model.v[:, 1])

        fig.canvas.draw()
        raw_input()
