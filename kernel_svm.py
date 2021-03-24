import numpy as np
from classes import Classifier
from utils import thresholding
from cvxopt import solvers, matrix
solvers.options['show_progress'] = False

class KernelSVM(Classifier): #wikipedia
    def __init__(self, C=1.0):
        self.C = C

    def fit(self, K, y):
        y = y*2 - 1
        assert(set(np.unique(y).tolist()) == {-1, +1})
        n = K.shape[0]
        dgf = np.diagflat(y).astype('float')
        P = matrix(K*np.outer(y, y))
        G = matrix(np.vstack([-np.eye(n), np.eye(n)]).astype('float'))
        h = matrix(np.hstack([np.zeros((n,), dtype=np.float), self.C*np.ones((n), dtype=np.float)]))
        A = matrix(y.astype('float').reshape((1, n)))
        b = matrix(0.0)
        result = solvers.qp(P=P, q=matrix(-np.ones((n,))), G=G, h=h, A=A, b=b)
        c = np.squeeze(np.array(result['x']))
        self.w = c*y
        idx = int(np.where(np.logical_and(c > 0, c < self.C))[0][0])
        self.beta = K[:, idx].dot(self.w) - y[idx]
        return self

    def predict(self, K):
        pred = K.dot(self.w) - self.beta
        return thresholding(pred, 0)