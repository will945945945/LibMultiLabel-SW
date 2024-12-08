import libmultilabel.linear as linear
import time
import numpy as np
import scipy.sparse as sparse
import argparse
import pickle
import os

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--datapath', type=str, default="")

ARGS = parser.parse_args()

print("start", flush=True)
start = time.time()
with open(ARGS.datapath + '.pkl', "rb") as F:
    datasets = pickle.load(F)
print("data loading cost:", time.time()-start, flush=True)

model_name = "Tree-with-all-labels_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        data = os.path.basename(ARGS.datapath)
        )

model_start = time.time()
model_name = "./models/" + model_name
if not os.path.isfile(model_name):
    model = linear.train_tree(
        datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -q", K=ARGS.K)
print("training cost:", time.time()-model_start, flush=True)

with open(submodel_name, "wb") as F:
    pickle.dump(model, F, protocol=5)

print("total cost:", time.time()-start, flush=True)
