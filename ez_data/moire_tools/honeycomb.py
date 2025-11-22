import numpy as np

def calc_wavelength(theta: float, a: float = 1.0, delta: float = 0.0) -> float:
    return a * (1 + delta)/np.sqrt(2 * (1 + delta) * (1 - np.cos(theta)) + delta**2)

def calc_density_from_wavelength(lam: float) -> float:
    return 2 / (3**0.5 * lam**2)

def calc_wavelength_from_density(n: float) -> float:
    return (2**0.5 / 2**0.25) / n**0.5

def calc_theta(lam: float, a: float, delta: float) -> float:
    return np.arccos(1 + 0.5 * delta / (1 + delta) - 0.5 * (1 + delta) * (a / lam)**2)

def calc_MSL_angle(theta: float, delta: float) -> float:
    return np.arctan(-np.sin(theta) / ((1 + delta) - np.cos(theta)))