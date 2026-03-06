import numpy as np

bounds_AA = np.array([[0, 1], [0, 0.1], [0, 1], [0, 0.5]])
bounds_AB = np.array([[0, 1], [0, 0.1], [0, 1], [0, 0.5], [0, 1], [0.5, 2]])
bounds_AC = np.array([[0, 1], [0, 0.1], [0, 1], [0, 0.5], [0, 1], [0.01, 4]])
bounds_AD = np.array([[0, 1], [0, 0.1], [0, 1], [0, 0.5], [0, 1], [0.5, 2], [0, 1], [0.01, 4]])

bounds_BA = np.array([[0, 1], [0, 0.1], [0, 1], [0.01, 0.4], [0, 1], [0, 0.5]])
bounds_BB = np.array([[0, 1], [0, 0.1], [0, 1], [0.01, 0.4], [0, 1], [0, 0.5], [0, 1], [0.5, 2]])
bounds_BC = np.array([[0, 1], [0, 0.1], [0, 1], [0.01, 0.4], [0, 1], [0, 0.5], [0, 1], [0.01, 4]])
bounds_BD = np.array([[0, 1], [0, 0.1], [0, 1], [0.01, 0.4], [0, 1], [0, 0.5], [0, 1], [0.5, 2], [0, 1], [0.01, 4]])

def perpendicular_model(t, *params):
    """
    Model describing the rotational autocorrelation function
    perpendicular to the principal axis of rotation, with a Gaussian
    """
    D_s, D_t = params
    c2 = 0.75 * np.exp(-t * (4 * D_s + 2 * D_t)) + 0.25 * np.exp(-t * 6 * D_t)
    return c2

def perpendicular_model_gauss(t, *params):
    """
    Model describing the rotational autocorrelation function
    perpendicular to the principal axis of rotation, with a Gaussian
    """
    D_s, D_t, A1, sigma = params
    c2 = 0.75 * np.exp(-t * (4 * D_s + 2 * D_t)) + 0.25 * np.exp(-t * 6 * D_t)
    gauss = np.exp(-0.5 * (t / sigma) ** 2)
    return A1 * c2 + (1 - A1) * gauss

def parallel_model_one_exp(t, *params):
    D_t, A1, tau1 = params
    c2 = np.exp(-t * D_t * 6)
    e1 = np.exp(-t / tau1)
    return A1 * c2 + (1 - A1) * e1

def parallel_model_one_exp_gauss(t, *params):
    D_t, A1, tau1, A2, sigma = params
    c2 = np.exp(-t * D_t * 6)
    e1 = np.exp(-t / tau1)
    gauss = np.exp(-0.5 * (t / sigma) ** 2)
    return A1 * c2 + A2 * e1 + (1 - A1 - A2) * gauss

def parallel_model_two_exp(t, *params):
    D_t, A1, tau1, A2, tau2 = params
    c2 = np.exp(-t * D_t * 6)
    e1 = np.exp(-t / tau1)
    e2 = np.exp(-t / tau2)
    return A1 * c2 + A2 * e1 + (1 - A1 - A2) * e2

def parallel_model_two_exp_gauss(t, *params):
    D_t, A1, tau1, A2, tau2, A3, sigma = params
    c2 = np.exp(-t * D_t * 6)
    e1 = np.exp(-t / tau1)
    e2 = np.exp(-t / tau2)
    gauss = np.exp(-0.5 * (t / sigma) ** 2)
    return A1 * c2 + A2 * e1 + A3 * e2 + (1 - A1 - A2 - A3) * gauss

def model_AA(t, *params):
    D_s, D_t, A2, tau2 = params
    parallel_out = parallel_model_one_exp(t, D_t, A2, tau2)
    perpendicular_out = perpendicular_model(t, D_s, D_t)
    return np.array([perpendicular_out, parallel_out])

def model_AB(t, *params):
    D_s, D_t, A2, tau2, A3, tau3 = params
    parallel_out = parallel_model_two_exp(t, D_t, A2, tau2, A3, tau3)
    perpendicular_out = perpendicular_model(t, D_s, D_t)
    return np.array([perpendicular_out, parallel_out])

def model_AC(t, *params):
    D_s, D_t, A2, tau2, A3, sigma2 = params
    parallel_out = parallel_model_one_exp_gauss(t, D_t, A2, tau2, A3, sigma2)
    perpendicular_out = perpendicular_model(t, D_s, D_t)
    return np.array([perpendicular_out, parallel_out])

def model_AD(t, *params):
    D_s, D_t, A2, tau2, A3, tau3, A4, sigma2 = params
    parallel_out = parallel_model_two_exp_gauss(t, D_t, A2, tau2, A3, tau3, A4, sigma2)
    perpendicular_out = perpendicular_model(t, D_s, D_t)
    return np.array([perpendicular_out, parallel_out])

def model_BA(t, *params):
    D_s, D_t, A1, sigma, A2, tau2 = params
    parallel_out = parallel_model_one_exp(t, D_t, A2, tau2)
    perpendicular_out = perpendicular_model_gauss(t, D_s, D_t, A1, sigma)
    return np.array([perpendicular_out, parallel_out])

def model_BB(t, *params):
    D_s, D_t, A1, sigma, A2, tau2, A3, tau3 = params
    parallel_out = parallel_model_two_exp(t, D_t, A2, tau2, A3, tau3)
    perpendicular_out = perpendicular_model_gauss(t, D_s, D_t, A1, sigma)
    return np.array([perpendicular_out, parallel_out])

def model_BC(t, *params):
    D_s, D_t, A1, sigma1, A2, tau2, A3, sigma2 = params
    parallel_out = parallel_model_one_exp_gauss(t, D_t, A2, tau2, A3, sigma2)
    perpendicular_out = perpendicular_model_gauss(t, D_s, D_t, A1, sigma1)
    return np.array([perpendicular_out, parallel_out])

def model_BD(t, *params):
    D_s, D_t, A1, sigma1, A2, tau2, A3, tau3, A4, sigma2 = params
    parallel_out = parallel_model_two_exp_gauss(t, D_t, A2, tau2, A3, tau3, A4, sigma2)
    perpendicular_out = perpendicular_model_gauss(t, D_s, D_t, A1, sigma1)
    return np.array([perpendicular_out, parallel_out])

bounds_list = [bounds_AA, bounds_AB, bounds_AC, bounds_AD, bounds_BA, bounds_BB, bounds_BC, bounds_BD] 
names = ["model_nogauss_oneexp", 'model_nogauss_twoexp', 'model_nogauss_oneexpgauss', 'model_nogauss_twoexpgauss', 'model_gauss_oneexp', 'model_gauss_twoexp', 'model_gauss_oneexpgauss', 'model_gauss_twoexpgauss']
perpendicular_models = [model_AA, model_AB, model_AC, model_AD, model_BA, model_BB, model_BC, model_BD]
param_names = {'model_nogauss_oneexp': ['D_s', 'D_t', 'A2', 'tau2'],
               'model_nogauss_twoexp': ['D_s', 'D_t', 'A2', 'tau2', 'A3', 'tau3'],
               'model_nogauss_oneexpgauss': ['D_s', 'D_t', 'A2', 'tau2', 'A3', 'sigma2'],
               'model_nogauss_twoexpgauss': ['D_s', 'D_t', 'A2', 'tau2', 'A3', 'tau3', 'A4', 'sigma2'],
               'model_gauss_oneexp': ['D_s', 'D_t', 'A1', 'sigma', 'A2', 'tau2'],
               'model_gauss_twoexp': ['D_s', 'D_t', 'A1', 'sigma', 'A2', 'tau2', 'A3', 'tau3'],
               'model_gauss_oneexpgauss': ['D_s', 'D_t', 'A1', 'sigma1', 'A2', 'tau2', 'A3', 'sigma2'],
               'model_gauss_twoexpgauss': ['D_s', 'D_t', 'A1', 'sigma1', 'A2', 'tau2', 'A3', 'tau3', 'A4', 'sigma2']}
