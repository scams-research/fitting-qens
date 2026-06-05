import numpy as np

def rotational_diffusion(q: np.ndarray, D_rot: float) -> np.ndarray:
    return np.ones_like(q) * D_rot


def lorentzian(x: np.ndarray, gamma: float, A: float)-> np.ndarray:
    x0 = 0
    return (A / np.pi) * (gamma / ((x - x0)**2 + gamma**2))