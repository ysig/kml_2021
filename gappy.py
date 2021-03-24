import numpy as np
from classes import Kernel
from training import train_evals
from argparser import parse
from collections import Counter
from itertools import combinations
from kernel_svm import KernelSVM
import functools
from scipy.sparse import csc_matrix


def get(d, elem):
    ret = d.get(elem, None)
    if ret is None:
        ret = len(d)
        d[elem] = ret
    return ret

class Gappy(Kernel):
    def __init__(self, g=8, l=6):
        self.g = g
        self.l = l
        self.num_features = None

    def fit_transform(self, X, X_test):
        X = X + X_test
        labels = {}
        rows, cols, data = [], [], []
        for i, s in enumerate(X):
            c = Counter()
            for k in range(len(s) - self.g+1):
                for gap in combinations(s[k:k+self.g], self.l):
                    # print(gap)
                    c[gap] += 1
            for k, v in c.items():
                idx = get(labels, k)
                rows.append(i)
                cols.append(idx)
                data.append(v)
        self.num_features = len(labels)
        Phi = csc_matrix((data, (rows, cols)), shape=(len(X), len(labels)), )
        kernel = Phi.dot(Phi.T).toarray().astype(np.float)
        return kernel

def gappy(dry):
    cp = {'C': 0.5}
    cps = [cp for _ in range(3)]
    train_evals(save_name=f"gappy_g8_l6", kernel=Gappy, classifier=KernelSVM, dry=dry, params_classifier=cps)

if __name__ == "__main__":
    gappy(parse().dry)