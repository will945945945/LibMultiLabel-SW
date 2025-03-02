#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os

# Add the parent directory to sys.path
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import pickle
import libmultilabel.linear as linear
import numpy as np
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from tqdm import tqdm
from libmultilabel.common_utils import AttributeDict
from scipy.special import log_expit, expit
from sklearn.model_selection import KFold


def l1_hinge_loss(x):
    """return max(0, 1 - x)"""
    return np.maximum(0, 1 - x)


def l2_hinge_loss(x):
    """return max(0, 1 - x)^2"""
    return np.maximum(0, 1 - x) ** 2


def decision_value_to_prob(decision_values, model_type, alpha=1.0):
    #eps: a scalar close to zero, which is used to avoid numerical issues when calculating cross entropy
    eps = np.finfo(decision_values.dtype).eps
    model_type = model_type.lower()

    loss_func = l2_hinge_loss if model_type == "l2svm" else l1_hinge_loss
    if model_type != "lr":
        prob = expit(-0.5 * alpha * (loss_func(decision_values) - loss_func(-decision_values)))
        return np.where(prob == 1, 1.0 - eps, prob)
    else:
        prob = expit(alpha * decision_values)
        return np.where(prob == 1, 1.0 - eps, prob)


def metrics_in_batches(model, batch_size, datasets, model_type, positive_label_idx, alpha):
    num_instances = datasets["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["CrossEntropy"], datasets["y"].shape[1])
    for i in range(num_batches):
        tmp_data = datasets["x"][i * batch_size : (i + 1) * batch_size]
        preds = model.predict_values(tmp_data)[:, positive_label_idx][:, np.newaxis]
        target = datasets["y"][i * batch_size : (i + 1) * batch_size].toarray()[:, positive_label_idx][:, np.newaxis]
        probs = decision_value_to_prob(preds, model_type, alpha)
        metrics.update(probs, target)
    metrics = metrics.compute()
    return metrics


data_names = ["a9a", "ijcnn1", "webspam", "real-sim", "rcv1"]
model_types = ["l2svm", "l1svm", "lr"]
modes = ["trvate", "trva"]
df_cols = "dataset,mode,model_type,tr_NLL,te_NLL,best_alpha".split(",")

pbar_dn = tqdm(data_names)
for dn in pbar_dn:
    pbar_dn.set_description(f"Dataset: {dn}")
    df = {_c: [] for _c in df_cols}
    for model_type in model_types:
        for mode in modes:

            # Load linear model
            logs_dir = f"../runs/{mode}"
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

            if model_type != "lr":
                # Grid search Alpha value
                _min = float("inf")
                best_tr = 0
                best_alpha = 0
                pbar_alpha = tqdm(np.arange(1, 10.1, 0.1))
                # Use Whole Training Set
                # ======================
                for alpha in pbar_alpha:
                    pbar_alpha.set_description(f"Alpha: {alpha}")
                    tmp_metrics = metrics_in_batches(
                        model, 2**16, datasets["train"], model_type, positive_label_idx, alpha
                    )["CrossEntropy"]
                    if tmp_metrics < _min:
                        _min = tmp_metrics
                        tr_metrics = tmp_metrics
                        best_alpha = alpha
                te_metrics = metrics_in_batches(
                    model, 2**16, datasets["test"], model_type, positive_label_idx, best_alpha
                )["CrossEntropy"]

                # Use 5-fold
                # ==========
                # kf = KFold(n_splits=5, shuffle=True, random_state=42)
                # for alpha in pbar_alpha:
                #     pbar_alpha.set_description(f"Alpha: {alpha}")
                #     avg_ce = 0

                #     for train_idx, val_idx in kf.split(datasets["train"]["x"]):
                #         val_subset = {"x":datasets["train"]["x"][val_idx], "y":datasets["train"]["y"][val_idx]}
                #         tmp_metrics = metrics_in_batches(model, 2**16, val_subset, model_type, positive_label_idx, alpha)['CrossEntropy']
                #         avg_ce += tmp_metrics / 5  # Compute average over 5 folds

                #     if avg_ce < _min:
                #         _min = avg_ce
                #         best_alpha = alpha

                # tr_metrics = metrics_in_batches(model, 2**16, datasets["train"], model_type, positive_label_idx, best_alpha)['CrossEntropy']
                # te_metrics = metrics_in_batches(model, 2**16, datasets["test"], model_type, positive_label_idx, best_alpha)['CrossEntropy']
            # lr
            else:
                tr_metrics = metrics_in_batches(model, 2**16, datasets["train"], model_type, positive_label_idx, 1.0)[
                    "CrossEntropy"
                ]
                te_metrics = metrics_in_batches(model, 2**16, datasets["test"], model_type, positive_label_idx, 1.0)[
                    "CrossEntropy"
                ]
                best_alpha = None
            tr_NLL = tr_metrics
            te_NLL = te_metrics
            for col in df_cols:
                df[col].append(eval(col) if col != "dataset" else eval("dn"))
    df = pd.DataFrame(df)
    df.to_csv(f"{dn}_alpha.csv", index=False)
