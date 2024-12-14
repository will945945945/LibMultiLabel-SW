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

parser.add_argument('--seed', type=int, default=27)
parser.add_argument('--K', type=int, default=10)
parser.add_argument('--beam_width', type=int, default=10000)
parser.add_argument('--num_models', type=int, default=3)
parser.add_argument('--sample_rate', type=float, default=0.1)
parser.add_argument('--datapath', type=str, default="")

ARGS = parser.parse_args()

print("start", flush=True)
start = time.time()
with open(ARGS.datapath + '.pkl', "rb") as F:
    datasets = pickle.load(F)
print("data loading cost:", time.time()-start, flush=True)

def metrics_in_batches(model_name, batch_size):
    num_instances = datasets["test"]["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["P@1", "P@3", "P@5"], num_classes=datasets["test"]["y"].shape[1])
    for i in range(num_batches):
        print("process batches id:", i, flush=True)
        tmp_data = datasets["test"]["x"][i * batch_size : (i + 1) * batch_size]
        total_preds = np.zeros([tmp_data.shape[0], datasets["train"]["y"].shape[1]], order='F')
        total_cnts = np.zeros(datasets["train"]["y"].shape[1])

        for model_idx in range(ARGS.num_models):
            print("model_idx:", model_idx)
            submodel_name = "./models/" + model_name + "-{}".format(model_idx)
            with open(submodel_name, "rb") as F:
                tmp = models, metalabels = pickle.load(F)

            for metalabel in tqdm(range(len(models))):
                model = models[metalabel][0]
                preds = model.predict_values(tmp_data, beam_width=ARGS.beam_width)
                preds = preds.toarray(order='F')
                indices = metalabels == metalabel
                total_preds[:, indices] += preds
                total_cnts[indices] += 1

        target = datasets["test"]["y"][i * batch_size : (i + 1) * batch_size].toarray()
        total_preds /= total_cnts+1e-16
        metrics.update(total_preds, target)

    return metrics.compute()

model_name = "Rand-label-Forest-No-replacement_{data}_seed={seed}_K={K}_sample-rate={sample_rate}.model".format(
        seed = ARGS.seed,
        K = ARGS.K,
        sample_rate = ARGS.sample_rate,
        data = os.path.basename(ARGS.datapath)
        )

metrics = metrics_in_batches(model_name, 2000)

print("mean in subsampled labels:", metrics)
print("Total time:", time.time()-start)
