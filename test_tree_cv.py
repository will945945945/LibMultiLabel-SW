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
C_range = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
metrics_for_eval = ["P@1", "P@3", "P@5"]
NUM_FOLDS = 5
BATCH_SIZE = 1000
def metrics_in_batches(model, data_x, data_y, batch_size, metrics_for_eval, prob_alpha):
    num_instances = data_x.shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = {}
    metrics.update({prob+str(alpha): linear.get_metrics(metrics_for_eval, num_classes=data_y.shape[1]) for prob, alpha in prob_alpha})
    for i in tqdm(range(num_batches)):
        batch_slice = np.s_[i * batch_size : (i + 1) * batch_size]
        model.predict_decision(data_x[batch_slice])
        target = data_y[batch_slice].toarray()
        for prob, alpha in prob_alpha:
            preds = model.predict_values(beam_width=ARGS.beamwidth, prob_type=prob, A=alpha)
            metrics[prob+str(alpha)].update(preds, target)
    return metrics

with open(ARGS.datapath, "rb") as F:
    datasets = pickle.load(F)

data_splits = []
for i in range(NUM_FOLDS):
    datapath = ARGS.dataname+"_"+str(i+1)+".pkl"
    with open(datapath, "rb") as f:
        data_splits.append(pickle.load(f))


probtype = {"l2": "exp-L2", "l1": "exp-L1", "lr": "sigmoid"}[ARGS.modeltype]
exp_key = probtype + str(1)
A_score = {C:{str(A):{metric:0. for metric in metrics_for_eval} for A in A_range} for C in C_range}
exp_score= {C:{metric:0. for metric in metrics_for_eval} for C in C_range}


def build_prob_A():
    pairs = [(prob, A) for prob in probtype_A for A in A_range]
    if (probtype, 1) not in pairs:
        pairs.append((probtype, 1))
    return pairs

# Cross Validation
for C in C_range:
    print(f"score of C = {C}")
    prob_A = build_prob_A()
    for i in range(NUM_FOLDS):
        modelpath = ARGS.modelname+"_c"+str(C)+"_5folds_"+str(i+1)+".pkl"
        with open(modelpath, "rb") as F:
            model = pickle.load(F)['model']


        data_y = data_splits[i]["train"]["y"]
        data_x = data_splits[i]["train"]["x"]
        metrics = metrics_in_batches(model, data_x, data_y, BATCH_SIZE, metrics_for_eval, prob_A)

        for A in A_range:
            eval = metrics[probtype_A[0] + str(A)].compute()
            for metric in metrics_for_eval:
                A_score[C][str(A)][metric] += eval[metric] / NUM_FOLDS
        eval = metrics[exp_key].compute()
        for metric in metrics_for_eval:
            exp_score[C][metric] += eval[metric] / NUM_FOLDS

    for A in A_range:
        msg = [k + f": {100*v:.2f}" for k, v in  A_score[C][str(A)].items()]
        print(f"score of A = {A}: " + " ".join(msg))

    msg = [k + f": {100*v:.2f}" for k, v in  exp_score[C].items()]
    print("score of exp: " + " ".join(msg))

 
#get best configurations for each metric
data_x = datasets["test"]["x"]
data_y = datasets["test"]["y"]
bests = {k: None for k in metrics_for_eval}
best_exp_C = {k: None for k in metrics_for_eval}
for metric in metrics_for_eval:
    re_organized_score = {(C, float(A)):A_score[C][A][metric] for C in C_range for A in A_score[C].keys()}
    re_organized_exp_score = {C:exp_score[C][metric] for C in C_range}
    print(metric, " : ", re_organized_score)
    bests[metric] = max(re_organized_score, key=re_organized_score.get)
    best_exp_C[metric] = max(re_organized_exp_score, key=re_organized_exp_score.get)
    print(metric, bests[metric])

print(bests)

eval_sigmoid = {metric:0. for metric in metrics_for_eval}
sigmoid_jobs = {}
#testing sigmoid_A
for metric in metrics_for_eval:
    C, A = bests[metric]
    sigmoid_jobs.setdefault(C, {}).setdefault(A, []).append(metric)
for C, metric_A in sigmoid_jobs.items():
    with open(ARGS.modelname+"_c"+str(C)+".pkl", "rb") as F:
        model = pickle.load(F)['model']
    prob_A = [("sigmoid", A) for A in metric_A]
    t = time.time()
    metrics = metrics_in_batches(model, data_x, data_y, BATCH_SIZE, metrics_for_eval, prob_A)
    print(f"predicition time for sigmoid A {time.time()-t:.2f} sec")
    
    for A, ms in metric_A.items():
        computed = metrics["sigmoid" + str(A)].compute()
        for metric in ms:
            eval_sigmoid[metric] = computed[metric]
            print(metric, " best C,A = ", (C, A))

print("sigmoid")
print("final scores:", eval_sigmoid)


eval_exp = {metric:0. for metric in metrics_for_eval}
exp_jobs = {}
#testing for prior works
for metric in metrics_for_eval:
    exp_jobs.setdefault(best_exp_C[metric], []).append(metric)
for C, ms in exp_jobs.items():
    with open(ARGS.modelname+"_c"+str(C)+".pkl", "rb") as F:
        model = pickle.load(F)['model']

    t = time.time()
    metrics = metrics_in_batches(model, data_x, data_y, BATCH_SIZE, metrics_for_eval, [(probtype, 1)])
    print(f"predicition time for exp {time.time()-t:.2f} sec")
    computed = metrics[exp_key].compute()
    for metric in ms:
        eval_exp[metric] = computed[metric]
        print(metric, "best C for exp = ", C)
print("exp")
print("final scores:", eval_exp)
