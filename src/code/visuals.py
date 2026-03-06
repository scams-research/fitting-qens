import numpy as np


def one_exp_decay(t, *params):
    """
    One exponential decay model.

    params = [A1, tau1]
    """
    A1, tau1 = params
    exp1 = A1 * np.exp(-t / tau1)
    return exp1


def one_exp_one_gauss_decay(t, *params):
    """
    One exponential plus one Gaussian decay model.

    params = [A1, tau1, A2, sigma]
    """
    A1, tau1, A2, sigma = params
    exp1 = A1 * np.exp(-t / tau1)
    gauss = A2 * np.exp(-0.5 * (t / sigma) ** 2)
    return exp1 + gauss, exp1, gauss


def two_exp_decay(t, *params):
    """
    Two exponential decay model.
    """
    A1, tau1, A2, tau2 = params
    exp1 = A1 * np.exp(-t / tau1)
    exp2 = A2 * np.exp(-t / tau2)
    return exp1 + exp2, exp1, exp2


bounds_two_exp_decay = [(0.05, 1), (0, 2), (0.05, 1), (2, 15)]


def two_exp_one_gauss_decay(t, *params):
    """
    Two exponential plus one Gaussian decay model.

    params = [A1, tau1, A2, tau2, sigma]
    """
    A1, tau1, A2, tau2, sigma = params
    exp1 = A1 * np.exp(-t / tau1)
    exp2 = A2 * np.exp(-t / tau2)
    gauss = (1 - (A1 + A2)) * np.exp(-0.5 * (t / sigma) ** 2)
    return exp1 + exp2 + gauss, exp1, exp2, gauss


def three_exp_decay(t, *params):
    """
    Three exponential decay model.

    params = [A1, tau1, A2, tau2, A3, tau3]
    """
    A1, tau1, A2, tau2, A3, tau3 = params
    exp1 = A1 * np.exp(-t / tau1)
    exp2 = A2 * np.exp(-t / tau2)
    exp3 = A3 * np.exp(-t / tau3)
    return exp1 + exp2 + exp3, exp1, exp2, exp3


def three_exp_one_gauss_decay(t, *params):
    """
    Three exponential plus one Gaussian decay model.

    params = [A1, tau1, A2, tau2, A3, tau3, sigma]
    """
    A1, tau1, A2, tau2, A3, tau3, sigma = params
    exp1 = A1 * np.exp(-t / tau1)
    exp2 = A2 * np.exp(-t / tau2)
    exp3 = A3 * np.exp(-t / tau3)
    gauss = (1 - (A1 + A2 + A3)) * np.exp(-0.5 * (t / sigma) ** 2)
    return exp1 + exp2 + exp3 + gauss, exp1, exp2, exp3, gauss


def model_sampler(samples, model_func, x):
    """
    samples: array of shape (n_samples, n_params)
    model_func: function that accepts (x, *params)
    x: array of x-values
    """
    # Transpose so we can unpack columns easily
    params = [samples[:, i] for i in range(samples.shape[1])]

    # Compute model for each row in samples
    curves = np.array([model_func(x, *param_set) for param_set in zip(*params)])

    return curves
