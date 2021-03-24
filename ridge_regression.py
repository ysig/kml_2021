import numpy as np
from training import train_evals
from utils import thresholding
from classes import Classifier
from argparser import parse

class RidgeRegression(Classifier):
    def __init__(self, reg=0.01):
        self.reg = reg

    def fit(self, X, y):
        self.beta = np.linalg.solve(X.T.dot(X) + self.reg*np.eye(X.shape[1]), X.T.dot(y))

    def predict(self, X):
        return thresholding(X.dot(self.beta))

def rr(dry):
    train_evals(RidgeRegression, 'ridge-regression', mat=True, dry=dry)

if __name__ == "__main__":
    rr(parse().dry)
