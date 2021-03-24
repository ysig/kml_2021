from training import train_evals
from gaussian import GaussianKernel
from argparser import parse
from kernel_svm import KernelSVM

def svm(dry):
    train_evals(KernelSVM, f'svm', GaussianKernel, mat=True, params_classifier=cps, dry=dry)

if __name__ == "__main__":
    svm(parse().dry)
    