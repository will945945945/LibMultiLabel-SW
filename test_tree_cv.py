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
probtype = ["sigmoid"] #, "exp-L1", "exp-L2"]
# probtype_l2 = [ "exp-L2","L2-prob"]
# probtype_l1 = ["exp-L1", "L1-prob",]
# probtype_lr = ["sigmoid"]
# probtype_A = ["L1-prob", "L2-prob"]
probtype_A = ["sigmoid"]
#probtype = ["L1-prob", "L2-prob"]
A_range = [1.,1.5,2.,2.5,3.,3.5,4.,5.,6.,7.,8.,9.,]
metrics_for_eval = ["P@1", "P@3", "P@5"]

def metrics_in_batches(model, batch_size, metric_for_eval, probtype, alpha = 0):
    num_instances = data_x.shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = {prob:linear.get_metrics(metric_for_eval, num_classes=data_y.shape[1]) for prob in probtype}
    metrics.update({prob+str(A): linear.get_metrics(metric_for_eval, num_classes=data_y.shape[1]) for prob in probtype_A for A in A_range})
    for i in tqdm(range(num_batches)):
        tmp_data = data_x[i * batch_size : (i + 1) * batch_size]
        model.predict_decision(tmp_data)
        target = data_y[i * batch_size : (i + 1) * batch_size].toarray()
        for prob in probtype:
            if (prob in probtype_A) and (alpha == 0):
                for A in A_range:
                    preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A=A)
                    print(preds.shape)
                    print(target.shape)
                    metrics[prob+str(A)].update(preds, target)
            else:
                if alpha != 0 :
                    preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A = alpha)
                else:
                    preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob,)
                metrics[prob].update(preds, target)
    return metrics

with open(ARGS.datapath, "rb") as F:
    datasets = pickle.load(F)

data_splits = []
for i in range(5):
    datapath = ARGS.dataname+"_"+str(i+1)+".pkl"
    with open(datapath, "rb") as f:
        data_splits.append(pickle.load(f))
# with open(ARGS.modelname, "rb") as F:
#     model = pickle.load(F)['model']

# if ARGS.modeltype == "l2":
#     probtype = probtype_l2
# elif ARGS.modeltype == "l1":
#     probtype = probtype_l1
# elif ARGS.modeltype == "lr":
#     probtype = probtype_lr

# Cross Validation
A_score = {str(A):{k:0. for k in ["P@1", "P@3", "P@5"]} for A in A_range}
for i in range(5):
    print("split", i)
    modelpath = ARGS.modelname+"_5folds_"+str(i+1)+".pkl"
    with open(modelpath, "rb") as F:
        model = pickle.load(F)['model']

    data_y = data_splits[i]["train"]["y"]
    data_x = data_splits[i]["train"]["x"]
    metrics = metrics_in_batches(model, 1000, metrics_for_eval, probtype)

    for prob in probtype:
        if prob in probtype_A:
            for A in A_range:
                eval = metrics[prob+str(A)].compute()
                A_score[str(A)]["P@1"] += eval["P@1"]/5
                A_score[str(A)]["P@3"] += eval["P@3"]/5
                A_score[str(A)]["P@5"] += eval["P@5"]/5
                print(prob+str(A))
                print("mean in subsampled labels:", eval)
        # else:
        #     eval = metrics[prob].compute()
        #     print(prob)
        #     print("mean in subsampled labels:", eval)
    
for A in A_range:
    msg = [k + f": {100*v:.2f}" for k, v in  A_score[str(A)].items()]
    print(f"score of A={A}: " + " ".join(msg))

best_A = {}
eval = {}
data_x = datasets["test"]["x"]
data_y = datasets["test"]["y"]
with open(ARGS.modelname+".pkl", "rb") as F:
     model = pickle.load(F)['model']
 
#testing
for key in ["P@1", "P@3", "P@5"]:
    re_organized_score = {A:A_score[A][key] for A in A_score.keys()}
    print(key, " : ", re_organized_score)
    best_A = float(max(re_organized_score, key=re_organized_score.get))

    t = time.time()
    metrics_for_eval = [key]
    metrics = metrics_in_batches(model, 1000, metrics_for_eval, probtype_A, best_A)
    print(f"predicition time {time.time()-t:.2f} sec")
    for prob in probtype_A:
        eval.update(metrics[prob].compute())
        print(key, " best A = ", best_A,)
        print("eval:", eval)
print("sigmoid")
print("final scores:", eval)

