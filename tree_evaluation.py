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

probtype_A = ["sigmoid"]
A_range = [1., 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5., 6., 7., 8., 9., 16., 32, 64]

metrics_for_eval = ["P@1", "P@3", "P@5"]#, "CrossEntropy"]

def metrics_in_batches(model, batch_size, metrics_for_eval, prob_alpha ): # prob_alpha = [(prob, alpha)]
    num_instances = data_x.shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = {}
    metrics.update({prob+str(alpha): linear.get_metrics(metrics_for_eval, num_classes=data_y.shape[1]) for prob, alpha in prob_alpha})

    for i in tqdm(range(num_batches)):
        tmp_data = data_x[i * batch_size : (i + 1) * batch_size]
        model.predict_decision(tmp_data)
        target = data_y[i * batch_size : (i + 1) * batch_size].toarray()
        for prob, alpha in prob_alpha:
            preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A=alpha)
            metrics[prob+str(alpha)].update(preds, target)
    return metrics

with open(ARGS.datapath, "rb") as F:
    datasets = pickle.load(F)

with open(ARGS.modelpath, "rb") as F:
    model = pickle.load(F)['model']


#for picking A
data_x = datasets["train"]["x"]
data_y = datasets["train"]["y"]
prob_A = [(prob, A) for prob in probtype_A for A in A_range]
t = time.time()
metrics = metrics_in_batches(model, 1000, metrics_for_eval, prob_A)
A_score = {str(A):{k:0. for k in metrics_for_eval} for A in A_range} 
print(f"predicition time {time.time()-t:.2f} sec")
for prob, A in prob_A:
    eval = metrics[prob+str(A)].compute()
    A_score[str(A)]["P@1"] += eval["P@1"]
    A_score[str(A)]["P@3"] += eval["P@3"]
    A_score[str(A)]["P@5"] += eval["P@5"]
    # A_score[str(A)]["CrossEntropy"] -= eval["CrossEntropy"]

for A in A_range:
    msg = [k + f": {100*v:.2f}" for k, v in  A_score[str(A)].items()]
    print(f"score of A = {A}: " + " ".join(msg))

if ARGS.modeltype == "l2":
    probtype = ["exp-L2", "sigmoid"]
elif ARGS.modeltype == "l1":
    probtype = ["exp-L1", "sigmoid"]
elif ARGS.modeltype == "lr":
    probtype = ["sigmoid"]

probtype = ["sigmoid"]
#set best_A to run on test
best_A = {}
eval = {}
data_x = datasets["test"]["x"]
data_y = datasets["test"]["y"]
prob_A = [(prob, 1.) for prob in probtype]
bests = {"P@1":0, "P@3":0, "P@5":0 }
for key in metrics_for_eval:
    re_organized_score = {A:A_score[A][key] for A in A_score.keys()}
    print(key, " : ", re_organized_score)
    best_A = float(max(re_organized_score, key=re_organized_score.get))
    if ("sigmoid", best_A) not in prob_A:
        prob_A.append(("sigmoid", best_A))
    bests[key] = best_A

#testing
metrics_for_eval = ["P@1", "P@3", "P@5"]
t = time.time()
metrics = metrics_in_batches(model, 1000, metrics_for_eval, prob_A)
print(f"predicition time {time.time()-t:.2f} sec")

#print output
print("sigmoid")
for key in metrics_for_eval:
    eval.update({key:metrics["sigmoid"+str(bests[key])].compute()[key]})
    print(key, " best A = ", bests[key],)
print("final scores:", eval)

# key = "CrossEntropy"
# eval = metrics["sigmoid"+str(bests[key])].compute()
# print("CrossEntropy", " best A = ", best_A,)
# print("final scores:", eval)

print("exp")
eval = metrics[probtype[0]+str(1.)].compute()
print("final scores:", eval)

print("A1 sigmoid")
eval = metrics["sigmoid"+str(1.)].compute()
print("final scores:", eval)
