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
parser.add_argument('--num_models', type=int, default=3)
parser.add_argument('--sample_rate', type=float, default=0.1)
parser.add_argument('--datapath', type=str, default="")
parser.add_argument('--idx', type=int, default=-1)

ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

print("start", flush=True)
start = time.time()
with open(ARGS.datapath + '.pkl', "rb") as F:
    datasets = pickle.load(F)
print("data loading cost:", time.time()-start, flush=True)

num_models = ARGS.num_models
seed_pool = []
while len(seed_pool) != num_models:
    seed = np.random.randint(2**31 - 1)
    if seed not in seed_pool:
        seed_pool += [seed]

model_idx = ARGS.idx
np.random.seed(seed_pool[model_idx])

model_name = "Rand-label-Forest_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

model_start = time.time()
model_name = "./models/" + model_name + "-{}".format(model_idx)
if not os.path.isfile(model_name):
    model, indices = linear.train_random_label_forests(
        datasets["train"]["y"], datasets["train"]["x"], "-s 1 -B 1 -e 0.0001 -m 32 -q", sample_rate=ARGS.sample_rate, K=ARGS.K)
    print("training cost:", time.time()-model_start, flush=True)
    with open(model_name, "wb") as F:
        pickle.dump((model, indices), F, protocol=5)

print("total cost:", time.time()-start, flush=True)
