import libmultilabel.linear as linear
import time
import numpy as np
import argparse
import pickle


parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default="")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument("--liblinear_options", type=str, default="-s 1 -B 1 -q")
parser.add_argument("--treepath", type=str, default="")
parser.add_argument("--buildtree", type=bool, default=False)
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)

with open(ARGS.datapath, "rb") as f:
    datasets = pickle.load(f)

t = time.time()
if(ARGS.buildtree):
    treeroot = linear.get_tree_structure(
        datasets["train"]["y"],
        datasets["train"]["x"],
        K=ARGS.K,
    )
    print(f"tree structuring time {time.time()-t:.2f} sec")
    with open(ARGS.treepath, "wb") as f:
        pickle.dump(treeroot, f)
else:
    with open(ARGS.treepath, "rb") as f:
        treeroot = pickle.load(f)   

t = time.time()
model = linear.train_tree(
    datasets["train"]["y"],
    datasets["train"]["x"],
    ARGS.liblinear_options,
    root=treeroot      
)
print(f"trainning time {time.time()-t:.2f} sec")

with open(ARGS.modelpath, "wb") as f:
    pickle.dump({"model": model},f)



