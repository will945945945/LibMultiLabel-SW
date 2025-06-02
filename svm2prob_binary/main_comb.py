import sys
import os

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import libmultilabel.linear as linear
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tqdm import tqdm
from libmultilabel.common_utils import AttributeDict
from scipy.special import expit


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
        if prob_type == "alpha" or prob_type == "franc":
            prob = expit(0.5 * alpha * (loss_func(-decision_values) - loss_func(decision_values)))
            return np.where(prob == 1, 1.0 - eps, prob)


def cal_metrics(preds, target, model_type, prob_type=None, alpha=None, A=None, B=None):
    metrics_ce = linear.get_metrics(["CrossEntropy"], 2)
    probs = decision_value_to_prob(preds, prob_type, model_type, alpha, A, B)
    # CrossEntropy
    metrics_ce.update(probs, target)
    metrics_ce = metrics_ce.compute()

    return metrics_ce["CrossEntropy"]

def find_alpha(model_type, prob_type, X, y, param, positive_label_idx):
    _min = float('inf')
    if prob_type == "franc":
        space = [1]
    else:
        space = np.arange(1, 10.1, 0.1)
    for alpha in space:
        cur_ce = 0
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in kf.split(X.toarray(), y.toarray()[:, positive_label_idx]):
            X_train, y_train = (X[train_idx], y[train_idx])
            X_test, y_test = (X[test_idx], y[test_idx])
            # Train the model
            model = linear.train_binary_and_multiclass(y_train, X_train, False, param)
            # Test Data
            decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
            target_test = y_test.toarray()[:, positive_label_idx][:, np.newaxis]
            ce = cal_metrics(
                decision_value_test, target_test, model_type, prob_type=prob_type, alpha=alpha
            )
            cur_ce += ce
        cur_ce /= 5
        if cur_ce < _min:
            _min = cur_ce
            best_alpha = alpha
    return _min, best_alpha
data_names = ["cod-rna", "covtype", "mushrooms"]
prob_types = ["alpha", "franc"]
model_types = ["l2svm", "l1svm", ]
df_cols = "dataset,model_type,te_NLL,alpha,A,B,best_C".split(",")

import sys
search = sys.argv[1]
if search == 'tuned':
    space = [i for i in range(-13, 11)]
else:
    space = [1]

model2s = {
    "l2svm":1,
    "l1svm":3,
    "lr":0
}
A, B = None, None
for prob_type in prob_types:
    for dn in data_names:
        df = {_c: [] for _c in df_cols}
        for model_type in model_types:
            ARGS = {
                    "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/trva.svm",
                    "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
                }
            ARGS = AttributeDict(ARGS)
            # Load Data
            datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
            preprocessor = linear.Preprocessor(False, False)
            datasets = preprocessor.fit_transform(datasets)
            positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]
            X, y = datasets["train"]["x"], datasets["train"]["y"]
            X_test, y_test = datasets["test"]["x"], datasets["test"]["y"]
            pbar_dn = tqdm(space)
            pbar_dn.set_description(f"Dataset: {dn}, Prob: {prob_type}, Model: {model_type}")
            best_C, best_alpha, best_ce = None, None, float('inf')
            for i in pbar_dn:
                C = 2 ** i 
                param = f"-s {model2s[model_type]} -c {C}"                
                ce, alpha = find_alpha(model_type, prob_type, X, y, param, positive_label_idx)
                if ce < best_ce:
                    best_ce = ce
                    best_alpha = alpha
                    best_C = C

            param = f"-s {model2s[model_type]} -c {best_C}" 
            te_NLL, alpha = find_alpha(model_type, prob_type, X, y, param, positive_label_idx)

            for col in df_cols:
                df[col].append(eval(col) if col != "dataset" else eval("dn"))

        df = pd.DataFrame(df)

        if search == "tuned":
            os.makedirs(f"tables/tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/tune/{prob_type}/{dn}.csv", index=False)
        else:
            os.makedirs(f"tables/no_tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/no_tune/{prob_type}/{dn}.csv", index=False)