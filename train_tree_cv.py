import libmultilabel.linear as linear
import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataname", type=str, default="")
parser.add_argument("--modelname", type=str, default="")
parser.add_argument("--liblinear_options", type=str, default="-s 1 -B 1 -q")
parser.add_argument("--treepath", type=str, default="")
parser.add_argument("--buildtree", type=bool, default=False)
parser.add_argument("--K", type=int, default=10)
parser.add_argument("--seed", type=int, default=1234)
ARGS = parser.parse_args()

np.random.seed(ARGS.seed)
data_splits = []
for i in range(5):
    datapath = ARGS.dataname+"_"+str(i+1)+".pkl"
    with open(datapath, "rb") as f:
        data_splits.append(pickle.load(f))
i_range = range(5)
#i_range = [2]
for i in i_range:
    data_y = sparse.vstack([data_splits[j]["train"]["y"] for j in range(5) if j != i])
    data_x = sparse.vstack([data_splits[j]["train"]["x"] for j in range(5) if j != i])

    t = time.time()
    if(ARGS.buildtree):
        treeroot = linear.get_tree_structure(
            data_y,
            data_x,
            K=ARGS.K,
        )
        print(f"tree structuring time {time.time()-t:.2f} sec")
        with open(ARGS.treepath+"_"+str(i+1)+".pkl", "wb") as f:
            pickle.dump(treeroot, f)
    else:
        with open(ARGS.treepath+"_"+str(i+1)+".pkl", "rb") as f:
            treeroot = pickle.load(f)   

    t = time.time()
    model = linear.train_tree(
        data_y,
        data_x,
        ARGS.liblinear_options,
        root=treeroot      
    )

    modelpath = ARGS.modelname+"_"+str(i+1)+".pkl"
    with open(modelpath, "wb") as f:
        pickle.dump({"model": model},f)

    print(f"trainning time {time.time()-t:.2f} sec")





