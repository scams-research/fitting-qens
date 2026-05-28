import sys
import pickle
import dynesty

from pathlib import Path

root = Path(__file__).resolve().parents[1].absolute()
utility_path = root / "utility"
sys.path.append(str(utility_path))

import paths
import numpy as np
from typing import List
from scipy.signal import fftconvolve
from scipy.special import spherical_jn  # j_l(QR)

from multiprocessing import Pool

sys.path.append(str(paths.code))
from plet_data import PletData

from bounds import D_fick, D_parr, D_perp, normaliser, u2, bkg
from data import data360, res360, data197, res197, q_mid360, q_mid197
from model import (
    lorentzian,
    fickian,
    rotational_diffusion,
    log_likelihood2,
    to_data_bundle,
    to_res_bundle,
)


def high_level_model(omega: np.ndarray, q: float, *params: List[float]):
    D_fick, D_parr, D_perp, r1, normaliser, u2 = params
    # hall ross
    amp_t = np.exp(-(q**2) * u2 / 3)
    gamma_fick = gamma_fick = fickian(q, D_fick)
    lorentzian_fick = lorentzian(omega, gamma_fick, amp_t)

    # Rotation
    amp1 = spherical_jn(1, q * r1) ** 2
    amp2 = spherical_jn(2, q * r1) ** 2

    gamma_perp = rotational_diffusion(q, D_perp)
    gamma_parr = rotational_diffusion(q, D_parr)

    l1 = lorentzian(omega, gamma_parr + gamma_perp, amp1)
    l2 = lorentzian(omega, 6 * gamma_perp, 1 / 4 * amp2) + lorentzian(
        omega, 2 * gamma_perp + 4 * gamma_parr, 3 / 4 * amp2
    )
    lorentzian_rotational1 = 3 * l1 + 5 * l2

    # mask for 0 point delta
    delta = spherical_jn(0, q * r1) ** 2
    domega = omega[1] - omega[0]
    delta_fn = np.zeros_like(omega)
    idx0 = np.argmin(np.abs(omega))
    delta_fn[idx0] = delta / domega

    # apply to lorentzian
    lorentzian_rotational1 = lorentzian_rotational1 + delta_fn

    return fftconvolve(lorentzian_rotational1, lorentzian_fick, mode="same")


def fixed_fick_rotate(omega: np.ndarray, q: float, *params: List[float]):
    D_fick, D_parr, D_perp, normaliser, u2, bkg = params

    r1 = 2.48
    HL = high_level_model(omega, q, D_fick, D_parr, D_perp, r1, normaliser, u2)

    return HL / normaliser + bkg


def fixed_fick_rotate_convolved(resolution: PletData, *params) -> np.ndarray:
    q_mid = resolution["q_mid"]
    omega_masked = resolution["omega_mid"]

    model = np.zeros((q_mid.size, omega_masked.size))
    omega_conv = np.linspace(-2, 2, (omega_masked.size * 2) - 1)
    q = 0
    n_global = 4
    global_params = params[0 : n_global - 1]
    u2 = params[n_global - 1]

    if resolution["energy"] == 3.60:
        q_params = params[n_global : n_global + 1 + q_mid.size]
        normaliser = q_params[0]
        bkg = q_params[1:]

    else:
        q_params = params[n_global + q_mid360.size + 1 :]
        normaliser = q_params[0]
        bkg = q_params[1:]

    for q in range(q_mid.size):
        model_unconvolved = fixed_fick_rotate(
            omega_conv, q_mid[q], *global_params, normaliser, u2, bkg[q]
        )
        model[q] = fftconvolve(
            resolution["masked"][q] / resolution["masked"][q].sum(),
            model_unconvolved,
            mode="valid",
        )
    return model


q_bounds = [bkg] * q_mid360.shape[0]
q_bounds197 = [bkg] * q_mid197.shape[0]

# Bundling for multicore running
data360_b = to_data_bundle(data360)
data197_b = to_data_bundle(data197)
res360_b = to_res_bundle(res360)
res197_b = to_res_bundle(res197)

res360_b["energy"] = 3.60  # force the tag
res197_b["energy"] = 1.97  # force the tag

bounds = (
    [D_fick, D_parr, D_perp]
    + [u2]
    + [normaliser]
    + q_bounds
    + [normaliser]
    + q_bounds197
)


model = fixed_fick_rotate_convolved


def prior_transform(u):
    """
    Transform parameters to the prior space.

    :param u: parameters in the prior space

    :return: transformed parameters
    """

    return [i * (b[1] - b[0]) + b[0] for i, b in zip(u, bounds)]


if __name__ == "__main__":
    nworkers = 4
    with Pool(processes=nworkers) as pool:
        sampler = dynesty.NestedSampler(
            log_likelihood2,
            prior_transform,
            len(bounds),
            logl_args=(data360_b, res360_b, data197_b, res197_b, model),
            pool=pool,
            queue_size=nworkers,
        )
        sampler.run_nested(print_progress=True)
        ns_res = sampler.results

    with open(str(paths.data) + "/pLET_aniso_model.pkl", "wb") as f:
        pickle.dump(ns_res.asdict(), f)
