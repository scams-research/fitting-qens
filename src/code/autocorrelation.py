import numpy as np
import matplotlib.pyplot as plt
import dynesty
import pickle
from autocorr_models import *

data = np.loadtxt('src/data/autocorrelation_samples.txt', skiprows=1)

perpendicular = data[1:-50, 0]
perpendicular_err = data[1:-50, 1]
parallel = data[1:-50, 2]
parallel_err = data[1:-50, 3]

t = np.linspace(0, 7.5, 150)[1:]

y = np.array([perpendicular, parallel])
err = np.array([perpendicular_err, parallel_err])

def log_likelihood(params, t, data, model, errors) -> float:
    """
    Calculate the log likelihood of the data given the model parameters.

    :param data: Data to fit.
    :param resolution: Resolution function data.
    :param params: Parameters for the model.
    :return: Log likelihood value.
    """
    model = model(t, *params)
    sigma2 = errors**2 
    return  -0.5 * np.sum((data - model) ** 2 / sigma2 + np.log(sigma2))

def prior_transform(u):
    """
    Transform parameters to the prior space.
    
    :param u: parameters in the prior space
    
    :return: transformed parameters
    """

    return [i * (b[1] - b[0]) + b[0] for i, b in zip(u, bounds)]

perpendicular_results = {}
for name, bounds, model in zip(names, bounds_list, perpendicular_models):
    print(f"Running {name} model")
    sampler = dynesty.DynamicNestedSampler(log_likelihood, prior_transform, len(bounds), 
                                    logl_args=(t, y, model, err), sample='rslice')

    sampler.run_nested(print_progress=True,nlive_init=1000, nlive_batch=500)

    ns_res = sampler.results

    with open(f'src/data/{name}.pkl', 'wb') as f:
        pickle.dump(ns_res.asdict(), f)
    perpendicular_results[name] = ns_res