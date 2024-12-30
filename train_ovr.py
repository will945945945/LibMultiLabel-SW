import libmultilabel.linear as linear
import time
import numpy as np
import argparse
import pickle


parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default="")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument("--liblinear_options", type=str, default="-s 1 -B 1 -q")
parser.add_argument("--seed", type=int, default=1234)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

with open(ARGS.datapath, "rb") as f:
    datasets = pickle.load(f)

t = time.time()
model = linear.train_1vsrest(
    datasets["train"]["y"],
    datasets["train"]["x"],
    ARGS.liblinear_options,   
)
print(f"trainning time {time.time()-t:.2f} sec")

with open(ARGS.modelpath, "wb") as f:
    pickle.dump({"model": model},f)



