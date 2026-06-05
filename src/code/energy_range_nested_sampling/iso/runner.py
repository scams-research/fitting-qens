import numpy as np
import scipp as sc
from typing import List
from scipy.signal import fftconvolve
from scipy.special import spherical_jn  # j_l(QR)
from pathlib import Path
import dynesty
import pickle
import sys


root = Path(__file__).resolve().parents[1].absolute()
utility_path = root / "utility"
sys.path.append(str(utility_path))
import paths
from model import lorentzian, rotational_diffusion, fickian

sys.path.append(str(paths.code))
from functions import MDANSEdata


def high_level_model(omega: np.ndarray, q: float, *params: List[float]):
    D_fick, D_rot, r1, normaliser, u2 = params
    # hall ross
    amp_t = np.exp(-(q**2) * u2 / 3)
    gamma_hr = gamma_hr = fickian(q, D_fick)
    lorentzian_hr = lorentzian(omega, gamma_hr, amp_t)

    # Rotation
    amp1 = spherical_jn(1, q * r1) ** 2
    amp2 = spherical_jn(2, q * r1) ** 2

    gamma_rotational1 = rotational_diffusion(q, D_rot)
    lorentzian_rotational1 = 3 * lorentzian(
        omega, gamma_rotational1, amp1
    ) + 5 * lorentzian(omega, 3 * gamma_rotational1, amp2)

    # mask find minimum at omega = 0
    delta = spherical_jn(0, q * r1) ** 2
    domega = omega[1] - omega[0]
    delta_fn = np.zeros_like(omega)
    idx0 = np.argmin(np.abs(omega))
    delta_fn[idx0] = delta / domega

    # apply to lorentzian
    lorentzian_rotational1 = lorentzian_rotational1 + delta_fn

    return fftconvolve(lorentzian_rotational1, lorentzian_hr, mode="same")


def fixed_fick_rotate(omega: np.ndarray, q: float, *params: List[float]):
    D_fick, D_rot, normaliser, u2, bkg = params
    r1 = 2.48

    HL = high_level_model(omega, q, D_fick, D_rot, r1, normaliser, u2)

    return HL / normaliser + bkg


def fixed_fick_rotate_unconvolved(q_mid, omega, *params) -> np.ndarray:
    model = np.zeros((q_mid.size, omega.size))
    q = 0
    # Params
    n_global = 4
    global_params = params[0:n_global]
    q_params = params[n_global : n_global + q_mid.size]
    for q in range(q_mid.size):
        model[q] = fixed_fick_rotate(omega, q_mid[q], *global_params, q_params[q])
    return model


def log_likelihood(params, data, q_mid, omega, model) -> float:
    model = model(q_mid, omega, *params)
    SQE = np.sum(((model - data) / (data)) ** 2)
    return -0.5 * SQE


def nll(params: List[float], data: MDANSEdata, q_mid, omega, model) -> float:
    return -log_likelihood(params, data, q_mid, omega, model)


def prior_transform(u):
    return [i * (b[1] - b[0]) + b[0] for i, b in zip(u, bounds)]

energies = [0.5, 0.75, 1.0, 1.25, 1.5]

for x in energies:
    sim290 = MDANSEdata(str(paths.data) + "/incoh_mdanse.csv")
    sim290.parse(energy_lim=x)
    sim290.scippbin(bins=12)
    q_mid = sc.midpoints(sim290.binned.coords["q"]).values
    omega = sim290.energy

    model_name = "double_bes"

    if model_name == "double_bes":
        bounds = [(0.01, 0.2), (0.01, 1), (0.01, 5000), (0.0001, 10)] + [
            (0.01, 30)
        ] * 12
        model = fixed_fick_rotate_unconvolved

    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        len(bounds),
        logl_args=(sim290.binned.values, q_mid, omega, model),
    )

    print(f"Running energy: {x}mev")
    sampler.run_nested(print_progress=True)

    ns_res = sampler.results

    code = f"{round(x * 100):03d}"
    with open(str(paths.data) + f"/iso_full_mod_{code}mev_run.pkl", "wb") as f:
        pickle.dump(ns_res.asdict(), f)
