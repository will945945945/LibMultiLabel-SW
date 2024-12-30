import libmultilabel.linear as linear
import time
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os
import time
import math
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default="")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument('--modeltype', type=str, default="l2")
ARGS = parser.parse_args()

with open(ARGS.datapath, "rb") as f:
    datasets = pickle.load(f)

with open(ARGS.modelpath, "rb") as f:
    model = pickle.load(f)['model']


num_instances = datasets["train"]["x"].shape[0]


data_x = datasets["train"]["x"]
data_y = datasets["train"]["y"]
preds = model[0].predict_values(data_x)

def cross_entropy(preds, y):
    num_instances = preds.shape[0]
    num_labels = y.shape[1]
    if ARGS.modeltype == "l2":
        preds = 1/(1 + np.exp(0.5 * 1 * (np.maximum(0, 1 - preds) ** 2 - np.maximum(0, 1 + preds) ** 2)))
    if ARGS.modeltype == "l1":
        preds = 1/(1 + np.exp(0.5 * 1 * (np.maximum(0, 1 - preds) - np.maximum(0, 1 + preds))))
    if ARGS.modeltype == "lr":
        preds = 1/(1 + np.exp(-preds))

    pos_preds = np.array( [ [ math.log(p) if p > 1e-12 else -100 for p in pred ] for pred in preds])
    neg_preds = np.array( [ [ math.log(1-p) if (1-p) > 1e-12 else -100 for p in pred ] for pred in preds])
    target = y.todense()
    
    loss = np.sum(np.multiply(pos_preds, target) + np.multiply(neg_preds, 1 - target), axis=0).ravel()
    return -loss
loss = cross_entropy(preds, data_y)
print("cross entropy loss:", loss)
print(np.max(loss))





