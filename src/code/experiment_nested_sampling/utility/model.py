import numpy as np
from typing import List, Callable
import sys
import paths

sys.path.append(str(paths.code))
from plet_data import PletData


def lorentzian(x: np.ndarray, gamma: float, A: float) -> np.ndarray:
    """
    Lorentzian function

    :param x: x values
    :param gamma: half width at half maximum
    :param A: amplitude

    :return: y values
    """
    x0 = 0

    return (A / np.pi) * (gamma / ((x - x0) ** 2 + gamma**2))


def fickian(q: np.ndarray, grad: float) -> np.ndarray:
    """
    Hall Ross function for QENS.

    :param x: q values (in inverse angstroms)
    :param tau: characteristic time (in ps)
    :param l: characteristic length (in angstroms)
    :return: Gamma values (in meV)
    """
    return grad * q**2


def rotational_diffusion(q: np.ndarray, D_rot: float) -> np.ndarray:
    """
    Rotational diffusion function for QENS.

    :param q: q values (in inverse angstroms)
    :param D_rot: rotational diffusion coefficient (in meV)
    :return: Gamma values (in meV)
    """
    return np.ones_like(q) * D_rot


def log_likelihood(
    data: PletData,
    resolution: PletData,
    model: Callable,
    params: List[float],
) -> float:
    """
    Calculate the log likelihood of the data given the model parameters.

    Parameters
    ----------
    data : PletData
    resolution : PletData
    res_test : array
        Resolution test normalization array (e.g. res_test360 or res_test197)
    model : callable
    params : list[float]

    Returns
    -------
    float
    """
    model_vals = model(resolution, *params)

    comp_data = data["masked"]
    comp_errors = data["errors"]

    sqe = np.sum(((model_vals - comp_data) / comp_errors) ** 2)

    return -0.5 * sqe


def nll(
    params: List[float],
    data360: PletData,
    resolution360: PletData,
    data197: PletData,
    resolution197: PletData,
    model,
) -> float:
    """
    Negative log likelihood function for optimization.

    :param params: Parameters for the model.
    :param data360: Data to fit at 3.60 Å⁻¹.
    :param resolution360: Resolution function data at 3.60 Å⁻¹.
    :param data197: Data to fit at 1.97 Å⁻¹.
    :param resolution197: Resolution function data at 1.97 Å⁻¹.
    :return: Negative log likelihood value.
    """
    return -log_likelihood2(
        params, data360, resolution360, data197, resolution197, model
    )


def log_likelihood2(
    params, data360, resolution360, data197, resolution197, model
) -> float:
    """
    Change the order of the parameters to match the expected order for dynesty.

    :param params: Parameters for the model.
    :param data: Data to fit.
    :param resolution: Resolution function data.
    :return: Log likelihood value.
    """
    return log_likelihood(data360, resolution360, model, params) + log_likelihood(
        data197, resolution197, model, params
    )


# Bundling for multicore running
def to_data_bundle(data_obj):
    return {
        "masked": np.asarray(data_obj.masked, dtype=float),
        "errors": np.asarray(data_obj.errors, dtype=float),
    }


def to_res_bundle(res_obj):
    return {
        "q_mid": np.asarray(
            res_obj.q_mid.values[np.invert(res_obj.data.masks["q"].values)], dtype=float
        ),
        "omega_mid": np.asarray(
            res_obj.omega_mid.values[np.invert(res_obj.data.masks["omega"].values)],
            dtype=float,
        ),
        "masked": np.asarray(res_obj.masked, dtype=float),
        "energy": float(res_obj.energy),
    }
