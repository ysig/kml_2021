# kernels-2021
Kaggle-Ranking of Kernels:

Linear-Regression: 0.60466
SVM: 0.55866
Kernel-Ridge-Regression: 0.61933
Spectrum_k6: 0.62600
Gappy_q8_l6: 0.62866
mismatch_m1_k8: 0.65400
mismatch_m1_k9: 0.67333

# Details about this code-base

Dependencies: cvxopt, scipy, numpy

You can run main.py as:
    `python main.py`
or with:
    `python main.py --dry`
for validation. In the latter case it needs `sciki-learn` for `train_test_split` and `accuracy_report`.
All the submitted kaggle files are in folder `predictions` and all the logs from validation are in the folder `logs`.
You can find the extensive evaluation logs with the most promising of the tested kernel parameters and C from 2^{-2}, ..., 2^{2} in 

The only kernel that is not in validation and in predictions and which has been implemented is the substring kernel, which
can be found at `substring.py`.

Files corresponding to kernels:
 - `spectrum.py`
 - `svm.py` and `gaussian.py`
 - `gappy.py`
 - `mismatch.py`
 - `substring.py`

Files corresponding to classifiers:
 - `kernel_svm.py`
 - `ridge_regression.py`
 - `kernel_ridge_regression.py`

Other files:
 - `training.py` corresponds to all the train and predict pipeline, for all methods.
 - `classes.py` used to define base classes for kernels and classifiers.
 - `argparser.py` used to define the simple command line argument of dry.
 - `utils.py` used to define a thresholding function.