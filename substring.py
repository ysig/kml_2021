import numpy as np
from classes import Kernel
from training import train_evals
from argparser import parse
from collections import Counter
from kernel_svm import KernelSVM
import functools


LAMDA = 0.8

@functools.lru_cache(maxsize=100000)
def B(x, xp, k):
    if k == 0:
        return 1
    elif min(len(x), len(xp)) < k:
        return 0
    else:
        out = B(x[:-1], xp, k)
        out += B(x, xp[:-1], k)
        out -= LAMDA*B(x[:-1], xp[:-1], k)
        if x[-1] == xp[-1]:
            out += LAMDA*B(x[:-1], xp[:-1], k-1)
        out *= LAMDA
        return out

@functools.lru_cache(maxsize=1024)
def K(x, xp, k):
    if min(len(x), len(xp)) < k:
        return 0
    else:
        # print('(x, xp, k)', (x, xp, k))
        out = 0.0
        cache = ""
        for a in xp:
            if a == x[-1] and len(cache) > 0:
                out += B(x[:-1], cache, k-1)
            cache += a
        out *= LAMDA**2
        out += K(x[:-1], xp, k)
        return out

def substring(x, xp, k):
    return sum([K(x, xp, i) for i in range(1, k + 1)])

class Substring(Kernel):
    def __init__(self, k=7):
        self.k = k

    def fit_transform(self, X, X_test):
        X = X + X_test
        K = np.zeros((len(X), len(X)), dtype=np.float)
        for i, x in enumerate(X):
            K[i, i] = substring(x, x, self.k)
            for j, xp in enumerate(X[i+1:], i+1):
                K[i, j] = substring(x, xp, self.k)
                K[j, i] = K[i, j]
        return K

if __name__ == "__main__":
    dry = parse().dry
    train_evals(save_name='substring', kernel=Substring, classifier=KernelSVM, dry=dry)
    # print(substring('science is organized knowledge','wisdom is organized life', 3))