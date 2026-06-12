import numpy as np
import argparse
import pickle
import math
import scipy.sparse as sparse

parser = argparse.ArgumentParser()

parser.add_argument("--datapath", type=str, default="")
parser.add_argument("--savingpath", type=str, default="")
parser.add_argument("--n_fold", type=int, default=5)
ARGS = parser.parse_args()


with open(ARGS.datapath, "rb") as f:
    datasets = pickle.load(f)

train_x = datasets["train"]["x"]
train_y = datasets["train"]["y"]

assert train_x.shape[0]==train_y.shape[0]
N = train_x.shape[0]
len_x = train_x.shape[1]
len_y = train_y.shape[1]
print(
    train_x.shape[0], train_x.shape[1],"\n",
    train_y.shape[0], train_y.shape[1]
    )


train = sparse.hstack((train_x, train_y))
index = np.arange(np.shape(train)[0])
np.random.shuffle(index)
train = train[index, :]

for i in range(ARGS.n_fold):
    start = math.ceil(i/ARGS.n_fold*N)
    end = math.ceil((i+1)/ARGS.n_fold*N)
    with open(ARGS.savingpath+"_"+str(i+1)+".pkl", "wb") as f:
        split_data = {"train":{'x':train[start:end, :len_x], 'y':train[start:end ,len_x:]}}
        print(split_data["train"]['x'].shape[1], split_data["train"]['y'].shape[1])
        assert split_data["train"]['x'].shape[1] == len_x
        assert split_data["train"]['y'].shape[1] == len_y
        pickle.dump(split_data, f)
