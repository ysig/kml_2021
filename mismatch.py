import numpy as np
from classes import Kernel
from training import train_evals
from argparser import parse
from collections import Counter
from kernel_svm import KernelSVM
import copy

class Trie(object):
    def __init__(self, parent=None, parent_edge_label=None):
        self.parent = parent
        self.children = {}
        self.pel = parent_edge_label
        if parent is not None:
            self.depth = parent.depth + 1
            self.id = parent.id + [self.pel]
            self.kmers = copy.deepcopy(parent.kmers)
            parent.children[self.pel] = self
        else:
            self.depth = 0
            self.id = []
            self.kmers = {}


    def dfs(self, X, l, m, k, km=None):
        if self.parent is None:
            # If root
            def obj(Xi):
                ln = len(Xi)-k+1
                return np.stack((np.arange(ln), np.zeros((ln,))), 1).astype(np.int)
            self.kmers = {i: obj(X[i]) for i in range(X.shape[0])}
            km = np.zeros((X.shape[0], X.shape[0]))
        else:
            kmers = {}
            for i, obj in self.kmers.items():
                # update the number of mismatches
                # a mismatch is when X[i, offset+depth-1] != connecting_edge_label
                update = np.where(X[i, obj[:, 0]+self.depth-1] != self.pel)
                obj[update, 1] += 1
                # keep only kmers with less than m mismatches
                entry = obj[obj[:, 1] <= m]
                if len(entry):
                    kmers[i] = entry
            self.kmers = kmers

        if len(self.kmers) > 0:
            if k == 0:
                # If we reach a leaf
                for i, a in self.kmers.items():
                    for j, b in self.kmers.items():
                        # explanation:
                        # starting from root these kmers have survived
                        # in a chain of k-letters (the core of which we match)
                        # and have at most m mismatches (as the others are discarded)
                        # thus all there combinations are matches.
                        km[i, j] += len(a)*len(b)
            else:
                for a in range(l):
                    # Create branch for every character in the alphabet
                    child = Trie(self, a)

                    # dfs recursion in child
                    km = child.dfs(X, l, m, k-1, km)

                    # if child empty - remove
                    if len(child.kmers) == 0:
                        del self.children[a]

        return km

class Mismatch(Kernel):
    def __init__(self, m=1, k=4):
        self.k = k
        self.m = m
        self.num_features = None

    def fit_transform(self, X, X_test):
        import datetime
        now = datetime.datetime.now()
        X = X + X_test
        alphabet = sorted(set(c for x in X for c in x))
        l = len(alphabet)

        # Convert to integers
        alphabet_dict = {a: i for i, a in enumerate(alphabet)}        
        # We keep only tose that have the smaller length
        len_min = min(len(x) for x in X)
        X_array = np.array([[alphabet_dict[c] for c in x][:len_min] for x in X])

        kernel = Trie().dfs(X_array, l=l, m=self.m, k=self.k)

        return kernel

def mismatch(dry):
    kp = {'k': 9, 'm':1}
    kps = [kp for _ in range(3)]
    train_evals(save_name=f'mismatch_m1_k9', kernel=Mismatch, classifier=KernelSVM, params_kernel=kps, dry=dry)


if __name__ == "__main__":
    mismatch(parse().dry)