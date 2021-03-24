def thresholding(X, thr=0.5):
    Xp = X.copy()
    Xp[X > thr] = 1
    Xp[X <= thr] = 0
    return Xp.astype(int)