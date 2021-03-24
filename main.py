from argparser import parse
from kernel_ridge_regression import krr
from ridge_regression import rr
from svm import svm
from spectrum import spectrum
from mismatch import mismatch
from gappy import gappy

if __name__ == "__main__":
    dry = parse().dry
    rr(dry)
    krr(dry)
    svm(dry)
    spectrum(dry)
    gappy(dry)
    mismatch(dry)
