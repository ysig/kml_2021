# sklearn is used only for utils
import os
import datetime
import numpy as np
from kernel_svm import KernelSVM

def read_mat(f):
    with open(f, 'r') as a:
        X = np.array([[float(i) for i in l.split(' ')] for l in a.readlines()])
    return X

def read_seq(f):
    with open(f, 'r') as a:
        X = [str(l.strip('\n').strip().split(',')[1]) for l in list(a.readlines())[1:]]
    return X

def read_label(f):
    with open(f, 'r') as a:
        X = np.array([int(l.strip('\n').strip().split(',')[1]) for l in list(a.readlines())[1:]])
    return X

def read_data(fold, train, mat):
    if train:
        if mat:
            X = read_mat(f'data/Xtr{fold}_mat100.csv')
        else:
            X = read_seq(f'data/Xtr{fold}.csv')
        y = read_label(f'data/Ytr{fold}.csv')
        return X, y
    else:
        if mat:
            X = read_mat(f'data/Xte{fold}_mat100.csv')
        else:
            X = read_seq(f'data/Xte{fold}.csv')
        return X

def save_preds(y_pred, save_name, fold):
    if not os.path.isdir('predictions'):
        os.mkdir('predictions')
    with open(f'predictions/{save_name}.csv', ('a' if fold > 0 else 'w')) as a:
        if fold == 0:
            a.write(f'Id,Bound\n')
        for j, y in enumerate(y_pred):
            i = fold*1000 + j
            a.write(f'{i},{y}\n')

def log_preds(y_test, y_pred, save_name, fold):
    from sklearn.metrics import classification_report
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    with open(f'logs/{save_name}.txt', ('a' if fold > 0 else 'w')) as a:
        print(f"fold #{fold}:\n{classification_report(y_test, y_pred)}", file=a)
        if fold == 0:
            np.save('temp.npy', (y_test, y_pred))
        elif fold == 1:
            yt1, y_pred1 = np.load('temp.npy')
            np.save('temp.npy', (np.hstack((yt1, y_test)), np.hstack((y_pred1, y_pred))))
        if fold == 2:
            yt1, y_pred1 = np.load('temp.npy')
            tst, pred = np.hstack((yt1, y_test)), np.hstack((y_pred1, y_pred))
            print(f"Total:\n{classification_report(tst, pred)}", file=a)
            os.remove('temp.npy')

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def is_kernel(x):
    eigmin = np.real(np.min(np.linalg.eigvals(x)))
    if eigmin < - 0.0001:
        raise Exception('Not PSD')
    elif not check_symmetric(x):
        raise Exception('Not Symmetric')

def train_eval(kernel, classifier, save_name, fold=0, mat=False, dry=False):
    time_start = datetime.datetime.now()
    X_train, y = read_data(fold, True, mat)
    if dry:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y, y_test = train_test_split(X_train, y, test_size=0.1, random_state=42, shuffle=True)
    else:
        X_test = read_data(fold, False, mat)

    if kernel is None:
        K_train = X_train
        classifier.fit(K_train, y)
        K_test = X_test
    else:
        K = kernel.fit_transform(X_train, X_test)
        if mat:
            classifier.num_features = X_train.shape[1]
        else:
            classifier.num_features = kernel.num_features
        
        K_train = K[:len(X_train), :len(X_train)]
        if not mat:
            # normalize
            X_diag = np.diag(K_train)
            K_train /= np.sqrt(np.outer(X_diag, X_diag))
        is_kernel(K_train)
        assert not np.any(np.isnan(K_train))
        classifier.fit(K_train, y)

        K_test = K[len(X_train):,:len(X_train)]
        if not mat:
            # normalize
            X_diag_test = np.diag(K[len(X_train):, len(X_train):])
            K_test /= np.sqrt(np.outer(X_diag_test, X_diag))
        assert not np.any(np.isnan(K_test))

    y_pred = classifier.predict(K_test)

    time_end = datetime.datetime.now()
    if dry:
        log_preds(y_test, y_pred, save_name, fold)
    else:
        save_preds(y_pred, save_name, fold)
    return time_end - time_start

def init_object(obj, p):
    if obj is None:
        return None
    if p is None:
        est = obj()
    elif isinstance(p, dict):
        est = obj(**p)
    elif isinstance(p, list):
        est = obj(*p)
    elif isinstance(p[0], tuple) and isintace(p[0], list) and isinstance(p[1], dict):
        est = obj(*p[0], **p[1])
    else:
        raise Exception('Wrong arguments.')
    return est

def make_time(delta):
    out = []
    minutes, seconds = divmod(delta.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        out.append(str(hours)+ "h")
    if minutes > 0:
        out.append(str(minutes)+"m")
    if seconds > 0:
        out.append(str(seconds)+"s")
    if minutes == 0:
        out.append(str(int(delta.microseconds/1000))+"ms")
    return ''.join(out)

def train_evals(classifier, save_name, kernel=None, params_classifier=None, params_kernel=None, mat=False, dry=True):
    if params_classifier is None:
        params_classifier = [None, None, None]
    if params_kernel is None:
        params_kernel = [None, None, None]

    times = datetime.datetime.now() 
    times = times - times
    for p, pp, fold in zip(params_kernel, params_classifier, [0, 1, 2]):
        krn = init_object(kernel, p)
        clf = init_object(classifier, pp)
        times += train_eval(krn, clf, save_name, fold, mat, dry)
    if not dry:
        with open('times_kaggle.txt', 'a') as f:
            tm = make_time(times/3)
            print(f"{save_name}: {tm}", file=f)

if __name__ == "__main__":
    from sklearn.svm import SVC
    train_evals(SVC, 'svc', mat=True, dry=True)