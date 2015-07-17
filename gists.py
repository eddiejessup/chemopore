from param_sweep import run_param_sweep
run_param_sweep(super_dirname='../data/free_space_rotational_diffusion',
                output_every=1000, t_upto=10.0, resume=True,
                tumbles=True, memorys=False,
                chis=0.0, phis=0.0, D_rot_0s=np.logspace(-2, 0.1, 4),
                seeds=[1, 2])
import glob;from param_sweep import measure_y_of_x
Dr_r, D_r, De_r = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*tumble-1*'),
                                 x_key='Dr', y_key='D')
Dr_a, D_a, De_a = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*tumble-0*'),
                                 x_key='Dr', y_key='D')


from param_sweep import run_param_sweep
run_param_sweep(super_dirname='../data/porous_diffusion',
                output_every=10000, t_upto=200.0, resume=True,
                tumbles=[True, False], memorys=False,
                chis=0.0, phis=np.linspace(0.0, 0.45, 22), D_rot_0s=1.0)
import glob;from param_sweep import measure_y_of_x
phi_r, D_r, De_r = measure_y_of_x(glob.glob('../data/porous_diffusion/*tumble-1*'),
                                  x_key='phi', y_key='D')
phi_a, D_a, De_a = measure_y_of_x(glob.glob('../data/porous_diffusion/*tumble-0*'),
                                  x_key='phi', y_key='D')


from param_sweep import run_param_sweep
run_param_sweep(super_dirname='../data/porous_rotational_diffusion',
                output_every=1000, t_upto=20.0, resume=True,
                tumbles=[False, True], memorys=False,
                chis=0.0, phis=0.24, D_rot_0s=np.logspace(-3, 0.2, 22),
                seeds=range(10),
                rho_0=5000.0)
import glob;from param_sweep import measure_y_of_x
Dr_r, D_r, De_r = measure_y_of_x(glob.glob('../data/porous_rotational_diffusion/*tumble-1*'),
                                 x_key='Dr', y_key='D')
Dr_a, D_a, De_a = measure_y_of_x(glob.glob('../data/porous_rotational_diffusion/*tumble-0*'),
                                 x_key='Dr', y_key='D')


from param_sweep import run_param_sweep
run_param_sweep(super_dirname='../data/free_space_drift_curve',
                output_every=10000, t_upto=100.01, resume=True,
                tumbles=[True, False], memorys=[True, False],
                chis=1.0 - np.logspace(-3, -0.01, 22), phis=0.0, D_rot_0s=1.0,
                seeds=0,
                rho_0=5000.0)
import glob;from param_sweep import measure_y_of_x
chi_r_s, vd_r_s, vde_r_s = measure_y_of_x(glob.glob('../data/free_space_drift_curve/*memory-0*tumble-1*'),
                                          x_key='chi', y_key='vd')
chi_a_s, vd_a_s, vde_a_s = measure_y_of_x(glob.glob('../data/free_space_drift_curve/*memory-0*tumble-0*'),
                                          x_key='chi', y_key='vd')
chi_r_t, vd_r_t, vde_r_t = measure_y_of_x(glob.glob('../data/free_space_drift_curve/*memory-1*tumble-1*'),
                                          x_key='chi', y_key='vd')
chi_a_t, vd_a_t, vde_a_t = measure_y_of_x(glob.glob('../data/free_space_drift_curve/*memory-1*tumble-0*'),
                                          x_key='chi', y_key='vd')
np.savez('free_space_drift_curve.npz',
         chi_r_s=chi_r_s, vd_r_s=vd_r_s, vde_r_s=vde_r_s,
         chi_a_s=chi_a_s, vd_a_s=vd_a_s, vde_a_s=vde_a_s,
         chi_r_t=chi_r_t, vd_r_t=vd_r_t, vde_r_t=vde_r_t,
         chi_a_t=chi_a_t, vd_a_t=vd_a_t, vde_a_t=vde_a_t)
d = np.load('free_space_drift_curve.npz')
(chi_r_s, vd_r_s, vde_r_s,
 chi_a_s, vd_a_s, vde_a_s,
 chi_r_t, vd_r_t, vde_r_t,
 chi_a_t, vd_a_t, vde_a_t) = (d['chi_r_s'], d['vd_r_s'], d['vde_r_s'],
                              d['chi_a_s'], d['vd_a_s'], d['vde_a_s'],
                              d['chi_r_t'], d['vd_r_t'], d['vde_r_t'],
                              d['chi_a_t'], d['vd_a_t'], d['vde_a_t'])
plt.errorbar(chi_a_t, vd_a_t, yerr=vde_a_t)
plt.errorbar(chi_a_s, vd_a_s, yerr=vde_a_s)
plt.errorbar(chi_r_t, vd_r_t, yerr=vde_r_t)
plt.errorbar(chi_r_s, vd_r_s, yerr=vde_r_s)


from param_sweep import run_param_sweep
run_param_sweep(super_dirname='../data/free_space_drift_rotational_diffusion',
                output_every=10000, t_upto=100.01, resume=True,
                tumbles=True, memorys=False,
                chis=0.476714354293, phis=0.0, D_rot_0s=np.logspace(-3, 0.2, 22),
                seeds=0,
                rho_0=5000.0)
run_param_sweep(super_dirname='../data/free_space_drift_rotational_diffusion',
                output_every=10000, t_upto=100.01, resume=True,
                tumbles=False, memorys=False,
                chis=0.477955417669, phis=0.0, D_rot_0s=np.logspace(-3, 0.2, 22),
                seeds=0,
                rho_0=5000.0)
run_param_sweep(super_dirname='../data/free_space_drift_rotational_diffusion',
                output_every=10000, t_upto=100.01, resume=True,
                tumbles=True, memorys=True,
                chis=0.911150709797, phis=0.0, D_rot_0s=np.logspace(-3, 0.2, 22),
                seeds=0,
                rho_0=5000.0)
run_param_sweep(super_dirname='../data/free_space_drift_rotational_diffusion',
                output_every=10000, t_upto=100.01, resume=True,
                tumbles=False, memorys=True,
                chis=0.976613898526, phis=0.0, D_rot_0s=np.logspace(-3, 0.2, 22),
                seeds=0,
                rho_0=5000.0)
import glob;from param_sweep import measure_y_of_x
Dr_r_s, D_r_s, De_r_s = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*memory-0*tumble-1*'),
                                       x_key='Dr', y_key='vd')
Dr_a_s, D_a_s, De_a_s = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*memory-0*tumble-0*'),
                                       x_key='Dr', y_key='vd')
Dr_r_t, D_r_t, De_r_t = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*memory-1*tumble-1*'),
                                       x_key='Dr', y_key='vd')
Dr_a_t, D_a_t, De_a_t = measure_y_of_x(glob.glob('../data/free_space_rotational_diffusion/*memory-1*tumble-0*'),
                                       x_key='Dr', y_key='vd')
