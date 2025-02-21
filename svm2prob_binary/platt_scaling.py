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

def sigmoid_predict(dec_value, A, B):
    fApB = dec_value * A + B
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

    prob = np.array([sigmoid_predict(float(x), A.value, B.value) for x in dec_values])
    return np.expand_dims(prob, axis=-1)


