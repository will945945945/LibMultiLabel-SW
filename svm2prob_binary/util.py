import numpy as np
from ctypes import *
import numpy as np
import libmultilabel.linear as linear
from sklearn.model_selection import KFold, StratifiedKFold

libsvm = CDLL('./libsvm.so')

# Define function signature
libsvm.sigmoid_train.argtypes = [
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
libsvm.sigmoid_train.restype = None

# Define function signature
libsvm.sigmoid_train_A.argtypes = [
    c_int,
    POINTER(c_double),
    POINTER(c_double),
    POINTER(c_double),
]
libsvm.sigmoid_train.restype = None

def sigmoid_predict(dec_value, A, B):
    fApB = dec_value * A + B
    if fApB >= 0:
        return np.exp(-fApB)/(1.0+np.exp(-fApB))
    else:
        return 1.0/(1+np.exp(fApB))

def sigmoid_predict_A(dec_value, A):
    fApB = dec_value * A
    if fApB >= 0:
        return np.exp(-fApB)/(1.0+np.exp(-fApB))
    else:
        return 1.0/(1+np.exp(fApB))

def sigmoid_train(dec_values, labels):
    l = len(dec_values)
    dec_values, labels = dec_values.squeeze(-1), labels.squeeze(-1)

    # Convert numpy arrays to C-compatible pointers
    dec_values_c = dec_values.astype(np.float64).ctypes.data_as(POINTER(c_double))
    labels_c = labels.astype(np.float64).ctypes.data_as(POINTER(c_double))
    
    # Create C doubles for A and B
    A = c_double()
    B = c_double()
    
    # Call the C function
    libsvm.sigmoid_train(l, dec_values_c, labels_c, byref(A), byref(B))

    return A.value, B.value

def sigmoid_train_A(dec_values, labels):
    l = len(dec_values)
    dec_values, labels = dec_values.squeeze(-1), labels.squeeze(-1)

    # Convert numpy arrays to C-compatible pointers
    dec_values_c = dec_values.astype(np.float64).ctypes.data_as(POINTER(c_double))
    labels_c = labels.astype(np.float64).ctypes.data_as(POINTER(c_double))
    
    # Create C doubles for A and B
    A = c_double()
    
    # Call the C function
    libsvm.sigmoid_train_A(l, dec_values_c, labels_c, byref(A))

    return A.value

def diff_term(model_type, norm, decision_values):
    if model_type == "l1svm":
        return np.where(1 - decision_values > 0, -decision_values / norm, 0)
    elif model_type == "l2svm":
        return np.where(1 - decision_values > 0, -2 * (1 - decision_values) * decision_values / norm, 0)
    else:
        return 0

def check_prob(model_type, norm, target, pos_probs, preds):
    return (-diff_term(model_type, norm, target * preds) + pos_probs * diff_term(model_type, norm, preds) + (1 - pos_probs) * diff_term(model_type, norm, -preds)).sum(0).squeeze()


def gen_S(X, y, positive_label_idx, param):
    n_samples = y.shape[0]
    decision_values = np.zeros(n_samples)

    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_idx, test_idx in kf.split(X):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        print(y[train_idx].toarray())
        for idx, (i, j) in enumerate(y[train_idx].toarray()):
            if i == j == 0:
                print(idx)
        model = linear.train_binary_and_multiclass(y_train, X_train, False, param)
        
        dec_vals = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]

        for i, idx in enumerate(test_idx):
            decision_values[idx] = dec_vals[i][0]
    return decision_values