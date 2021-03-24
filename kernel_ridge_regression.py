import numpy as np
from classes import Classifier
from training import train_evals
from utils import thresholding
from argparser import parse
from gaussian import GaussianKernel

class KernelRidgeRegression(Classifier):
    def __init__(self, reg=0.01):
        self.reg = reg

    def fit(self, X, y):
        y = y*2 - 1
        assert(set(np.unique(y).tolist()) == {-1, +1})
        self.alpha = np.linalg.solve(X + self.reg*self.num_features*np.eye(X.shape[0]), y)

    def predict(self, X):
        return thresholding(X.dot(self.alpha), 0)

def krr(dry):
    train_evals(KernelRidgeRegression, 'kernel-ridge-regression', GaussianKernel, mat=True, dry=dry)

if __name__ == "__main__":
    krr(parse().dry)
