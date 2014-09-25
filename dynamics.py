from runner import get_filenames, filename_to_model
import numpy as np
from scipy.stats import sem


def particle_dynamics(dr, dt):
    # Drift speed
    v_drift = dr / dt
    v_drift_mean = np.mean(v_drift, axis=0)
    v_drift_err = sem(v_drift, axis=0)

    # Drift-corrected diffusivity along each dimension
    dr_diffusive = (dr - np.mean(dr, axis=0))
    dr_sq_diffusive = np.square(dr_diffusive)
    D = dr_sq_diffusive / (2.0 * dt)
    D_mean = np.mean(D, axis=0)
    D_err = sem(D, axis=0)

    # Drift-corrected total diffusivity across all dimensions
    D_total = np.sum(D, axis=-1)
    D_total_mean = np.mean(D_total, axis=0)
    D_total_err = sem(D_total, axis=0)
    return D_mean, D_err, v_drift_mean, v_drift_err, D_total_mean, D_total_err


def model_to_dr_dt(model_1, model_0):
    dr = model_1.get_unwrapped_r() - model_0.get_unwrapped_r()
    dt = model_1.t - model_0.t
    return dr, dt


def system_evolution(output_dirname):
    output_filenames = get_filenames(output_dirname)
    first_output_filename = output_filenames[0]
    first_model = filename_to_model(first_output_filename)
    v_drifts, v_drifts_err = [], []
    Ds, Ds_err = [], []
    D_totals, D_totals_err = [], []
    for output_filename in output_filenames:
        model = filename_to_model(output_filename)
        dr, dt = model_to_dr_dt(model, first_model)
        (D_mean, D_err, v_drift_mean, v_drift_err,
         D_total_mean, D_total_err) = particle_dynamics(dr, dt)

        v_drifts.append(v_drift_mean)
        v_drifts_err.append(v_drift_err)
        Ds.append(D_mean)
        Ds_err.append(D_err)
        D_totals.append(D_total_mean)
        D_totals_err.append(D_total_err)
    return (np.array(Ds), np.array(Ds_err),
            np.array(v_drifts), np.array(v_drifts_err),
            np.array(D_totals), np.array(D_totals_err))
