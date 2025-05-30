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
parser.add_argument('--dataname', type=str, default="")
parser.add_argument('--beamwidth', type=int, default=10)
parser.add_argument('--modelname', type=str, default="")
parser.add_argument('--modeltype', type=str, default="l2")

ARGS = parser.parse_args()
full_preds = []
# probtype = ["exp-L1", "exp-L2","hardtanh","sigmoid","square-like-sigmoid"]
# probtype = ["sigmoid"] #, "exp-L1", "exp-L2"]
# probtype_l2 = [ "exp-L2","L2-prob"]
# probtype_l1 = ["exp-L1", "L1-prob",]
# probtype_lr = ["sigmoid"]
# probtype_A = ["L1-prob", "L2-prob"]
probtype_A = ["sigmoid"]
#probtype = ["L1-prob", "L2-prob"]
A_range = [0.25, 0.5, 1., 1.5, 2., 2.5, 3., 4., 5., 6., 7., 8., 10., 12., 16.]
metrics_for_eval = ["P@1", "P@3", "P@5"]

def metrics_in_batches(models, batch_size, metrics_for_eval, prob_alpha):
    num_instances = data_x.shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = {}
    metrics.update({prob+str(alpha): linear.get_metrics(metrics_for_eval, num_classes=data_y.shape[1]) for prob, alpha in prob_alpha})
    for i in tqdm(range(num_batches)):
        tmp_data = data_x[i * batch_size : (i + 1) * batch_size]
        target = data_y[i * batch_size : (i + 1) * batch_size].toarray()
        preds_per_config = {}
        for model in models:
            model.predict_decision(tmp_data)
            for prob, alpha in prob_alpha:
                preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A=alpha)
                if prob+str(alpha) not in preds:
                    preds_per_config[prob+str(alpha)] = preds
                else:
                    preds_per_config[prob+str(alpha)] += preds
        
        for prob, alpha in prob_alpha:
            metrics[prob+str(alpha)].update(preds_per_config[prob+str(alpha)]/3, target)

    return metrics

with open(ARGS.datapath, "rb") as F:
    datasets = pickle.load(F)

data_splits = []
for i in range(5):
    datapath = ARGS.dataname+"_"+str(i+1)+".pkl"
    with open(datapath, "rb") as f:
        data_splits.append(pickle.load(f))

# Cross Validation
prob_A = [(prob, A) for prob in probtype_A for A in A_range]

if ARGS.modeltype == "l2":
    prob_A.append(("exp-L2", 1)) 
    # prob_A = [("exp-L2", 1)]
    probtype = ["exp-L2"]
elif ARGS.modeltype == "l1":
    prob_A.append(("exp-L1", 1)) 
    # prob_A = [("exp-L1", 1)]
    probtype = ["exp-L1"]
elif ARGS.modeltype == "lr":
    prob_A.append(("sigmoid", 1)) 
    # prob_A = [("exp-L1", 1)]
    probtype = ["sigmoid"]
A_score = {str(A):{k:0. for k in metrics_for_eval} for A in A_range} 
A_score['exp'] = {k:0. for k in metrics_for_eval}
# A_score = {'exp': {k:0. for k in metrics_for_eval}}
for i in range(5):
    modelpath = ARGS.modelname+"_5folds_"+str(i+1)+".pkl"
    models = []
    with open(modelpath, "rb") as F:
        models.append(pickle.load(F)['model'])

    for seed in ['1235', '1236']:
        with open(modelpath.replace('1234', seed), "rb") as F:
            models.append(pickle.load(F)['model'])
    
    data_y = data_splits[i]["train"]["y"]
    data_x = data_splits[i]["train"]["x"]
    metrics = metrics_in_batches(models, 2000, metrics_for_eval, prob_A)

    for A in A_range:
        eval = metrics[probtype_A[0]+str(A)].compute()
        A_score[str(A)]["P@1"] += eval["P@1"]/5
        A_score[str(A)]["P@3"] += eval["P@3"]/5
        A_score[str(A)]["P@5"] += eval["P@5"]/5

    eval = metrics[probtype[0]+str(1)].compute()
    A_score["exp"]["P@1"] += eval["P@1"]/5
    A_score["exp"]["P@3"] += eval["P@3"]/5
    A_score["exp"]["P@5"] += eval["P@5"]/5

for A in A_range:
    msg = [k + f": {100*v:.2f}" for k, v in  A_score[str(A)].items()]
    print(f"score of A = {A}: " + " ".join(msg))

msg = [k + f": {100*v:.2f}" for k, v in  A_score['exp'].items()]
print("score of exp: " + " ".join(msg))

best_A = {}
eval = {}
data_x = datasets["test"]["x"]
data_y = datasets["test"]["y"]
models = []
with open(ARGS.modelname+".pkl", "rb") as F:
        models.append(pickle.load(F)['model'])
for seed in ['1235', '1236']:
    with open(ARGS.modelname.replace('1234', seed)+".pkl", "rb") as F:
        models.append(pickle.load(F)['model'])

#testing
best_A = {}
eval = {}
data_x = datasets["test"]["x"]
data_y = datasets["test"]["y"]
prob_A = []
bests = {"P@1":0, "P@3":0, "P@5":0 }
for key in metrics_for_eval:
    re_organized_score = {A:A_score[A][key] for A in A_score.keys() if A != 'exp'}
    print(key, " : ", re_organized_score)
    best_A = float(max(re_organized_score, key=re_organized_score.get))
    if ("sigmoid", best_A) not in prob_A:
        prob_A.append(("sigmoid", best_A))
    bests[key] = best_A

if ARGS.modeltype == "l2":
    prob_A.append(("exp-L2", 1)) 
    # prob_A = [("exp-L2", 1)]
    probtype = ["exp-L2"]
elif ARGS.modeltype == "l1":
    prob_A.append(("exp-L1", 1)) 
    # prob_A = [("exp-L1", 1)]
    probtype = ["exp-L1"]
elif ARGS.modeltype == "lr":
    prob_A.append(("sigmoid", 1)) 
    # prob_A = [("exp-L1", 1)]
    probtype = ["sigmoid"]

#testing
metrics_for_eval = ["P@1", "P@3", "P@5"]
t = time.time()
metrics = metrics_in_batches(models, 2000, metrics_for_eval, prob_A)
print(f"predicition time {time.time()-t:.2f} sec")

print("sigmoid")
for key in metrics_for_eval:
    eval.update({key:metrics["sigmoid"+str(bests[key])].compute()[key]})
    print(key, " best A = ", bests[key])
print("final scores:", eval)

print("exp")
eval = metrics[probtype[0]+str(1)].compute()
print("final scores:", eval)