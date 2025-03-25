import numpy as np
from ctypes import *
from ctypes.util import find_library
from os import path
from glob import glob
import sys

try:
    import numpy as np
    import scipy
    from scipy import sparse
except:
    scipy = None

libsvm = CDLL('../libsvm/libsvm.so')

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
