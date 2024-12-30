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

parser.add_argument('--datapath', type=str, default="")
parser.add_argument('--beamwidth', type=int, default=10)
parser.add_argument('--modelpath', type=str, default="")
parser.add_argument('--modeltype', type=str, default="l2")

ARGS = parser.parse_args()
full_preds = []
# probtype = ["exp-L1", "exp-L2","sigmoid-1","sigmoid-2","sigmoid-3","sigmoid-4","sigmoid-5","hardtanh","square-like-sigmoid", "L1-prob", "L2-prob"]
probtype_l2 = [ "exp-L2","L2-prob"]
probtype_l1 = ["exp-L1", "L1-prob",]
probtype_lr = ["sigmoid"]
probtype_A = ["L1-prob", "L2-prob"]
#probtype = ["L1-prob", "L2-prob"]
A_range = np.arange(1,4,0.5)

def metrics_in_batches(model, batch_size, probtype,):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = {prob:linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1]) for prob in probtype}
    metrics.update({prob+str(A): linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1]) for prob in probtype_A for A in A_range})
    for i in tqdm(range(num_batches)):
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
        model.predict_decision(tmp_data)
        for prob in probtype:
            if prob in probtype_A:
                for A in A_range:
                    preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A=A)
                    target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
                    metrics[prob+str(A)].update(preds, target)
            else:
                preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob)
                target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
                metrics[prob].update(preds, target)
    return metrics

with open(ARGS.datapath, "rb") as F:
    datasets = pickle.load(F)

with open(ARGS.modelpath, "rb") as F:
    model = pickle.load(F)['model']

if ARGS.modeltype == "l2":
    probtype = probtype_l2
elif ARGS.modeltype == "l1":
    probtype = probtype_l1
elif ARGS.modeltype == "lr":
    probtype = probtype_lr

t = time.time()
metrics = metrics_in_batches(model, 1000, probtype)
print(f"beam_search time {time.time()-t:.2f} sec")
for prob in probtype:
    if prob in probtype_A:
        for A in A_range:
            eval = metrics[prob+str(A)].compute()
            print(prob+str(A))
            print("mean in subsampled labels:", eval)
    else:
        eval = metrics[prob].compute()
        print(prob)
        print("mean in subsampled labels:", eval)



