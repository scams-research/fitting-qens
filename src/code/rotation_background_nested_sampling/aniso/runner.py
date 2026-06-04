import numpy as np
import scipp as sc
from typing import List
from scipy.special import spherical_jn  # j_l(QR)
from pathlib import Path
import dynesty
import pickle
import sys


root = Path(__file__).resolve().parents[1].absolute()
utility_path = root / "utility"
sys.path.append(str(utility_path))
import paths
from model import lorentzian, rotational_diffusion

sys.path.append(str(paths.code))
from functions import MDANSEdata


rot = MDANSEdata(str(paths.data) + "/rotation_only_incoh_mdanse.csv")
rot.parse(energy_lim=0.4)
rot.scippbin(bins=12)
q_bins = 12
rot.binned = rot.binned["q", :q_bins]

# Delta peak removal
for q in range(q_bins):
    max_point = np.where(rot.binned.values[q] == rot.binned.values[q].max())[0]
    rot.binned.values[q][max_point] = rot.binned.values[q][max_point - 1]


bkg_number = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
for number in reversed(bkg_number):

    def fixed_fick_rotate(omega: np.ndarray, q: float, *params: List[float]):
        D_parr, D_perp, mult = params

        r1 = 2.48
        amp11 = spherical_jn(1, q * r1) ** 2
        amp12 = spherical_jn(2, q * r1) ** 2

        gamma_perp = rotational_diffusion(q, D_perp)
        gamma_parr = rotational_diffusion(q, D_parr)
        # 1st order bessel term:
        l1 = lorentzian(omega, gamma_parr + gamma_perp, amp11)
        l2 = lorentzian(omega, 6 * gamma_perp, 1 / 4 * amp12) + lorentzian(
            omega, 2 * gamma_perp + 4 * gamma_parr, 3 / 4 * amp12
        )
        return mult * (l1 + 5 / 3 * l2)

    def with_bkg(omega, q, *params):
        D_parr, D_perp, mult, bkg = params
        return fixed_fick_rotate(omega, q, *params[0:3]) + params[3]

    def fixed_fick_rotate_unconvolved(q_mid, omega, *params) -> np.ndarray:
        model = np.zeros((q_mid.size, omega.size))
        q = 0
        global_params = params[0:3]
        bkg = params[3:]
        for i in range(q_mid.size):
            if i >= number:
                model[q] = with_bkg(omega, q_mid[q], *global_params, bkg[q - number])
                q += 1
            else:
                model[q] = fixed_fick_rotate(omega, q_mid[q], *global_params)
                q += 1
        return model

    def log_likelihood(params, data, q_mid, omega, model) -> float:
        model = model(q_mid, omega, *params)
        SQE = np.sum(((model - data) / (data)) ** 2)
        return -0.5 * SQE

    def nll(params: List[float], data: MDANSEdata, q_mid, omega, model) -> float:
        return -log_likelihood(params, data, q_mid, omega, model)

    def prior_transform(u):
        return [i * (b[1] - b[0]) + b[0] for i, b in zip(u, bounds)]

    np.random.seed(42)
    model_name = "bkg"

    if model_name == "bkg":
        bounds = [(0.1, 1), (0.01, 0.06), (0.001, 10)] + [(0.001, 3)] * (12 - number)
        model = fixed_fick_rotate_unconvolved

    q_mid = sc.midpoints(rot.binned.coords["q"]).values
    omega = rot.energy

    print(f"number_bkg = {12 - number}")
    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        len(bounds),
        logl_args=(rot.binned.values, q_mid, omega, model),
    )

    sampler.run_nested(print_progress=True)

    ns_res = sampler.results
    nbkg = 12-number

    with open(
        str(paths.data) + f"/rotation_only_aniso_model_nbkg{nbkg:02d}.pkl", "wb"
    ) as f:
        pickle.dump(ns_res.asdict(), f)
