import sys
import pickle


from pathlib import Path

root = Path(__file__).resolve().parents[1].absolute()
utility_path = root / "utility"
sys.path.append(str(utility_path))

import paths
import numpy as np
from scipy.signal import fftconvolve
from dynesty.utils import Results
from runner import fixed_fick_rotate
from data import res360, res197, q_mid360, q_mid197

sys.path.append(str(paths.code))
from plet_data import PletData


def fixed_fick_rotate_sample(resolution: PletData,q, *params) -> np.ndarray:
    q_mid = resolution.q_mid.values[np.invert(resolution.data.masks['q'].values)]
    omega_masked = resolution.omega_mid.values[np.invert(resolution.data.masks['omega'].values)]

    omega_conv = np.linspace(-2, 2, (omega_masked.size * 2) - 1)

    n_global = 3
    global_params = params[0:n_global]
    q_params = params[n_global:n_global+q_mid.size+1]
    model_unconvolved = fixed_fick_rotate(omega_conv, q_mid[q], *global_params, q_params[0], q_params[q+1])


    res = resolution.masked


    model = fftconvolve(res[q] / res[q].sum(), model_unconvolved, mode='valid')
    return model


def full_model_sampler(resolution, chosen_q, samples):
    results = []
    for i, params in enumerate(samples):
        model = fixed_fick_rotate_sample(resolution, chosen_q, *params)
        results.append(model)
    return np.stack(results, axis=0)


with open(str(paths.data) + "/pLET_iso_model.pkl", "rb") as f:
    ns_res = pickle.load(f)
ns_res = Results(ns_res)


n_global = 3
ns_res_samples360 = ns_res.samples_equal()[:, : n_global + 1 + q_mid360.shape[0]]
ns_res_samples197 = np.hstack(
    [
        ns_res.samples_equal()[:, 0:n_global],
        ns_res.samples_equal()[:, n_global + 1 + q_mid360.shape[0] :],
    ]
)

samples360 = np.zeros(
    [q_mid360.size, ns_res_samples360.shape[0], res360.masked.shape[1]]
)
samples197 = np.zeros(
    [q_mid197.size, ns_res_samples197.shape[0], res197.masked.shape[1]]
)


for q in range(q_mid360.size):
    samples360[q] = full_model_sampler(res360, q, ns_res_samples360)
for q in range(q_mid197.size):
    samples197[q] = full_model_sampler(res197, q, ns_res_samples197)

with open(str(paths.data) + "pLET_iso_model_samples_197.pkl", "wb") as handle:
    pickle.dump(samples197, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(str(paths.data) + "pLET_iso_model_samples_360.pkl", "wb") as handle:
    pickle.dump(samples360, handle, protocol=pickle.HIGHEST_PROTOCOL)
