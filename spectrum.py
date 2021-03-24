import numpy as np
from classes import Classifier
from training import train_evals
from classes import Kernel
from argparser import parse
from scipy.sparse import csc_matrix
from collections import Counter
from kernel_svm import KernelSVM

def add_to_dict(a, d):
    if a not in d:
        d[a] = len(d)
    return d[a]

class Spectrum(Kernel):
    def __init__(self, k=6):
        self.k = k

    def fit_transform(self, X, X_test):
        X = X + X_test
        labels = dict()
        rows, cols, data = [], [], []
        for idx, x in enumerate(X):
            cnt = Counter()
            for i in range(len(x)):
                cnt[add_to_dict(x[i], labels)] += 1
                for k in range(1, self.k+1):
                    if i < len(x) - k:
                        cnt[add_to_dict(x[i:i+k+1], labels)] += 1
            for i, c in cnt.items():
                rows.append(idx)
                cols.append(i)
                data.append(c)
        self.num_features = len(labels)
        Phi = csc_matrix((data, (rows, cols)), shape=(len(X), len(labels)), )
        return Phi.dot(Phi.T).toarray().astype(np.float)

def spectrum(dry):
    train_evals(save_name=f'Spectrum_k6', kernel=Spectrum, classifier=KernelSVM, dry=dry)

if __name__ == "__main__":
    spectrum(parse().dry)