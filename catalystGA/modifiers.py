import numpy as np


def gaussian(x, target, sigma):
    return np.exp(-0.5 * np.power((x - target) / sigma, 2))
