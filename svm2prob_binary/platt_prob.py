import sys
import os

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import pickle
import libmultilabel.linear as linear
import numpy as np
import time
import math
import pandas as pd

from tqdm import tqdm
from libmultilabel.common_utils import AttributeDict
from platt_scaling import sigmoid_train, sigmoid_predict


def decision_value_to_prob(decision_values, A, B):
    #eps: a scalar close to zero, which is used to avoid numerical issues when calculating cross entropy
    eps = np.finfo(decision_values.dtype).eps
    prob = np.expand_dims(np.array([sigmoid_predict(float(x), A, B) for x in decision_values.squeeze(-1)]), axis=-1)
    return np.where(prob == 1, 1.0 - eps, prob)


def metrics_in_batches(model, batch_size, datasets, positive_label_idx, A, B):
    num_instances = datasets["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)
    metrics = {"platt": linear.get_metrics(["CrossEntropy"], datasets["y"].shape[1])}
    for i in range(num_batches):
        tmp_data = datasets["x"][i * batch_size : (i + 1) * batch_size]
        preds = model.predict_values(tmp_data)[:, positive_label_idx][:, np.newaxis]
        target = datasets["y"][i * batch_size : (i + 1) * batch_size].toarray()[:, positive_label_idx][:, np.newaxis]
        probs = decision_value_to_prob(preds, A, B)
        metrics["platt"].update(probs, target)
    metrics["platt"] = metrics["platt"].compute()
    return metrics


data_names = ["a9a", "ijcnn1", "webspam", "real-sim", "rcv1", "rcv1_reverse"]
model_types = ["l2svm", "l1svm"]
modes = ["trvate", "trva"]
df_cols = "dataset,mode,model_type,tr_NLL,te_NLL,A,B".split(",")
import sys
root = sys.argv[1]
for dn in tqdm(data_names):
    df = {_c: [] for _c in df_cols}
    for model_type in model_types:
        for mode in modes:
            # Load linear model and dataset.
            logs_dir = f"{root}/{mode}"
            model_path_prefix = f"{dn}_{model_type}_c"
            model_path = sorted([os.path.join(logs_dir, _d) for _d in os.listdir(logs_dir) if _d.startswith(model_path_prefix)])[-1]
            ARGS = {
                "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/{mode}.svm",
                "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
                "modelpath": f"{model_path}/linear_pipeline.pickle",
            }
            ARGS = AttributeDict(ARGS)

            datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
            preprocessor = linear.Preprocessor(False, False)
            datasets = preprocessor.fit_transform(datasets)
            positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]

            with open(ARGS.modelpath, "rb") as F:
                model = pickle.load(F)["model"]

            # Train Platt model.
            preds = model.predict_values(datasets["train"]["x"])[:, positive_label_idx][:, np.newaxis]
            target = datasets["train"]["y"].toarray()[:, positive_label_idx][:, np.newaxis]
            A, B = sigmoid_train(preds, target)

            # Calculate and evaluate probability.
            tr_metrics = metrics_in_batches(model, 2**16, datasets["train"], positive_label_idx, A, B)
            te_metrics = metrics_in_batches(model, 2**16, datasets["test"], positive_label_idx, A, B)
            tr_NLL = tr_metrics["platt"]["CrossEntropy"]
            te_NLL = te_metrics["platt"]["CrossEntropy"]
            for col in df_cols:
                df[col].append(eval(col) if col != "dataset" else eval("dn"))
    df = pd.DataFrame(df)
    if root == "../models/runs_tuned":
        df.to_csv(f"tables/tune/platt/{dn}_platt.csv", index=False)
    else:
        df.to_csv(f"tables/no_tune/platt/{dn}_platt.csv", index=False)
