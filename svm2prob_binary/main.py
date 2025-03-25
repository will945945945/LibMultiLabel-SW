import sys
import os
import json

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import pickle
import libmultilabel.linear as linear
import numpy as np
import math
import pandas as pd

from tqdm import tqdm
from libmultilabel.common_utils import AttributeDict
from scipy.special import expit
from util import sigmoid_train_A, sigmoid_predict_A, sigmoid_train, sigmoid_predict, check_prob


def l1_hinge_loss(x):
    """return max(0, 1 - x)"""
    return np.maximum(0, 1 - x)


def l2_hinge_loss(x):
    """return max(0, 1 - x)^2"""
    return np.maximum(0, 1 - x) ** 2


def decision_value_to_prob(decision_values, prob_type, model_type, alpha=None, A=None, B=None):
    # eps: a scalar close to zero, which is used to avoid numerical issues when calculating cross entropy
    eps = np.finfo(decision_values.dtype).eps
    model_type = model_type.lower()

    loss_func = l2_hinge_loss if model_type == "l2svm" else l1_hinge_loss

    if model_type == "lr":
        prob = expit(decision_values)
        return np.where(prob == 1, 1.0 - eps, prob)
    else:
        if prob_type.startswith("alpha_") or prob_type == "franc":
            prob = expit(0.5 * alpha * (loss_func(-decision_values) - loss_func(decision_values)))
            return np.where(prob == 1, 1.0 - eps, prob)
        if prob_type == "platt_onlyA":
            eps = np.finfo(decision_values.dtype).eps
            prob = np.expand_dims(
                np.array([sigmoid_predict_A(float(x), A) for x in decision_values.squeeze(-1)]), axis=-1
            )
            return np.where(prob == 1, 1.0 - eps, prob)
        if prob_type == "platt":
            eps = np.finfo(decision_values.dtype).eps
            prob = np.expand_dims(
                np.array([sigmoid_predict(float(x), A, B) for x in decision_values.squeeze(-1)]), axis=-1
            )
            return np.where(prob == 1, 1.0 - eps, prob)


def metrics_in_batches(model, batch_size, datasets, model_type, positive_label_idx, prob_type=None, alpha=None, A=None, B=None):
    num_instances = datasets["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)

    metrics = linear.get_metrics(["CrossEntropy"], datasets["y"].shape[1])

    res = 0
    for i in range(num_batches):
        tmp_data = datasets["x"][i * batch_size : (i + 1) * batch_size]
        preds = model.predict_values(tmp_data)[:, positive_label_idx][:, np.newaxis]
        target = datasets["y"][i * batch_size : (i + 1) * batch_size].toarray()[:, positive_label_idx][:, np.newaxis]
        probs = decision_value_to_prob(preds, prob_type, model_type, alpha, A, B)
        res += check_prob(model_type, np.linalg.norm(model.weights), target, probs, preds)
        metrics.update(probs, target)
    metrics = metrics.compute()
    return metrics["CrossEntropy"], res


data_names = ["a9a", "ijcnn1", "webspam", "real-sim", "rcv1", "rcv1_reverse"]
prob_types = ["franc", "alpha_ce", "alpha_diff", "platt", "platt_onlyA"]
model_types = ["l2svm", "l1svm", "lr"]
modes = ["trvate", "trva"]
df_cols = "dataset,mode,model_type,tr_NLL,te_NLL,tr_diff,te_diff,alpha,A,B".split(",")
import sys

root = sys.argv[1]

pbar_dn = tqdm(prob_types)
for prob_type in pbar_dn:
    pbar_dn.set_description(f"Dataset: {prob_type}")
    for dn in data_names:
        df = {_c: [] for _c in df_cols}
        for model_type in model_types:
            for mode in modes:
                # Load linear model
                logs_dir = f"{root}/{mode}"
                model_path_prefix = f"{dn}_{model_type}_c"
                model_path = sorted(
                    [os.path.join(logs_dir, _d) for _d in os.listdir(logs_dir) if _d.startswith(model_path_prefix)]
                )[-1]
                ARGS = {
                    "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/{mode}.svm",
                    "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
                    "modelpath": f"{model_path}/linear_pipeline.pickle",
                    "log": f"{model_path}/logs.json",
                }
                ARGS = AttributeDict(ARGS)

                datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
                preprocessor = linear.Preprocessor(False, False)
                datasets = preprocessor.fit_transform(datasets)
                positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]

                with open(ARGS.modelpath, "rb") as F:
                    model = pickle.load(F)["model"]

                with open(ARGS.log, "rb") as F:
                    C = float(json.load(F)["config"]["liblinear_options"].split(" ")[-1])

                lamda_tau = None
                alpha = None
                A = None
                B = None
                selection = prob_type.split("_")[1] if prob_type.startswith("alpha_") else None

                if model_type == "lr":
                    lamda_tau = 2 / C * np.linalg.norm(model.weights)

                if prob_type.startswith("alpha_"):
                    # Grid search Alpha value
                    _min = float("inf")
                    best_tr = 0
                    best_alpha = 0
                    # Use Whole Training Set
                    # ======================
                    for tmp_alpha in np.arange(1, 10.1, 0.1):
                        tmp_metrics, tmp_res = metrics_in_batches(
                            model, 2**16, datasets["train"], model_type, positive_label_idx, prob_type=prob_type, alpha=tmp_alpha
                        )
                        lamda_tau = 2 / C * np.linalg.norm(model.weights) / tmp_alpha
                        tmp_diff = abs(tmp_res - lamda_tau)
                        if selection != "ce":
                            metric = tmp_diff
                        else:
                            metric = tmp_metrics
                        if metric < _min:
                            _min = metric
                            best_alpha = tmp_alpha

                    lamda_tau = 2 / C * np.linalg.norm(model.weights) / best_alpha
                    alpha = best_alpha

                if prob_type == "platt":
                    # Train Platt model.
                    preds = model.predict_values(datasets["train"]["x"])[:, positive_label_idx][:, np.newaxis]
                    target = datasets["train"]["y"].toarray()[:, positive_label_idx][:, np.newaxis]

                    lamda_tau = 2 / C * np.linalg.norm(model.weights)
                    A, B = sigmoid_train(preds, target)

                if prob_type == "platt_onlyA":
                    # Train Platt model (No B).
                    preds = model.predict_values(datasets["train"]["x"])[:, positive_label_idx][:, np.newaxis]
                    target = datasets["train"]["y"].toarray()[:, positive_label_idx][:, np.newaxis]

                    lamda_tau = 2 / C * np.linalg.norm(model.weights)
                    A = sigmoid_train_A(preds, target)

                if prob_type == "franc":
                    lamda_tau = 2 / C * np.linalg.norm(model.weights)
                    alpha = 1

                tr_metrics, tr_res = metrics_in_batches(
                    model, 2**16, datasets["train"], model_type, positive_label_idx, prob_type=prob_type, alpha=alpha, A=A, B=B
                )
                te_metrics, te_res = metrics_in_batches(
                    model, 2**16, datasets["test"], model_type, positive_label_idx, prob_type=prob_type, alpha=alpha, A=A, B=B
                )
                tr_NLL = tr_metrics
                te_NLL = te_metrics
                tr_diff = abs(tr_res - lamda_tau)
                te_diff = abs(te_res - lamda_tau)
                for col in df_cols:
                    df[col].append(eval(col) if col != "dataset" else eval("dn"))

        df = pd.DataFrame(df)

        if root == "../models/runs_tuned":
            os.makedirs(f"tables/tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/tune/{prob_type}/{dn}.csv", index=False)
        else:
            os.makedirs(f"tables/no_tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/no_tune/{prob_type}/{dn}.csv", index=False)
