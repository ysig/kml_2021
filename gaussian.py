import numpy as np
from scipy.spatial.distance import cdist
from classes import Kernel

def gaussian_kernel(X, Xp, var=1.0):
    dist = np.square(cdist(X, Xp, metric='euclidean'))
    return np.exp(-dist/var)

class GaussianKernel(Kernel):
    def __init__(self, var=None):
        self.var = var
    def fit_transform(self, X, X_test):
        if self.var is None:
            self.var = X.shape[1] * X.var()
        x = np.vstack([X, X_test])
        return gaussian_kernel(x, x, self.var)