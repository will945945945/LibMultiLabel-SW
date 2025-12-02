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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def l1_hinge_loss(x):
    """return max(0, 1 - x)"""
    return np.maximum(0, 1 - x)


def l2_hinge_loss(x):
    """return max(0, 1 - x)^2"""
    return np.maximum(0, 1 - x) ** 2


def decision_value_to_prob(decision_values, model_type, alpha=None):
    eps = np.finfo(decision_values.dtype).eps
    model_type = model_type.lower()

    loss_func = l2_hinge_loss if model_type == "l2svm" else l1_hinge_loss

    if model_type == "lr":
        prob = expit(decision_values)
        return np.where(prob == 1, 1.0 - eps, prob)
    else:
        prob = expit(0.5 * alpha * (loss_func(-decision_values) - loss_func(decision_values)))
        return np.where(prob == 1, 1.0 - eps, prob)


def cal_metrics(preds, target, model_type, prob_type=None, alpha=None):
    metrics_ce = linear.get_metrics(["CrossEntropy"], 2)
    probs = decision_value_to_prob(preds, model_type, alpha)
    metrics_ce.update(probs, target)
    metrics_ce = metrics_ce.compute()
    return metrics_ce["CrossEntropy"]

def find_alpha(model_type, X, y, param, positive_label_idx):
    ce_alpha = float('inf')
    for alpha in [i / 10 for i in range(10, 101)]:
        cur_ce = 0
        kf = StratifiedKFold(n_splits=5, shuffle=False)
        for train_idx, test_idx in kf.split(X.toarray(), y.toarray()[:, positive_label_idx]):
            X_train, y_train = (X[train_idx], y[train_idx])
            X_test, y_test = (X[test_idx], y[test_idx])
            # Train the model
            model = linear.train_binary_and_multiclass(y_train, X_train, False, param)
            # Test Data
            decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
            target_test = y_test.toarray()[:, positive_label_idx][:, np.newaxis]
            ce = cal_metrics(
                decision_value_test, target_test, model_type, alpha=alpha
            )
            cur_ce += ce
        cur_ce /= 5
        if cur_ce < ce_alpha:
            ce_alpha = cur_ce
            best_alpha = alpha
        if alpha == 1:
            ce_franc = cur_ce
        
    return {
        "alpha": (ce_alpha, best_alpha),
        "franc": (ce_franc, 1.0),
    }

data_names = [
    "a9a", "ijcnn1", "rcv1", "real-sim", "webspam",
    "a1a", "a2a", "a3a", "a4a", "a5a", "a6a", "a7a", "a8a",
    "breast-cancer_scale", "ionosphere_scale", "diabetes_scale",
    "liver-disorders",
    "madelon",
    "sonar_scale", "gisette_scale",
    "skin_nonskin", "phishing", "mushrooms",
]

prob_types = ["alpha", "franc"]
model_types = ["lr", "l2svm", "l1svm"]
df_cols = "dataset,model_type,te_NLL,alpha,best_C".split(",")

search = sys.argv[1]
if search == "tuned":
    space = [i for i in range(-13, 11)]
else:
    space = [0]

model2s = {
    "l2svm": 1,
    "l1svm": 3,
    "lr": 0,
}

for dn in data_names:
    results = {pt: {c: [] for c in df_cols} for pt in prob_types}
    ARGS = {
        "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/trva.svm",
        "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
    }
    ARGS = AttributeDict(ARGS)
    datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
    preprocessor = linear.Preprocessor(False, False)
    datasets = preprocessor.fit_transform(datasets)

    try:
        positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]
    except Exception:
        positive_label_idx = np.where(preprocessor.label_mapping == 2)[0][0]

    X, y = datasets["train"]["x"], datasets["train"]["y"]
    X_test, y_test = datasets["test"]["x"], datasets["test"]["y"]

    for model_type in model_types:
        best_ce = {pt: float("inf") for pt in prob_types}
        best_C = {pt: None for pt in prob_types}
        best_alpha = None

        pbar_dn = tqdm(space)
        pbar_dn.set_description(f"Dataset: {dn}, Model: {model_type}")

        for i in pbar_dn:
            C = 2 ** i
            param = f"-s {model2s[model_type]} -c {C}"

            ce_dict = find_alpha(
                model_type,
                X,
                y,
                param,
                positive_label_idx,
            )

            for pt in prob_types:
                if ce_dict[pt][0] < best_ce[pt]:
                    best_ce[pt] = ce_dict[pt][0]
                    best_C[pt] = C
                    if pt == "alpha":
                        best_alpha = ce_dict[pt][1]
                    
        unique_Cs = sorted(set(best_C.values()))
        full_eval = {}

        for C in unique_Cs:
            param = f"-s {model2s[model_type]} -c {C}"
            model = linear.train_binary_and_multiclass(y, X, False, param)
            # Test Data
            decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
            target_test = y_test.toarray()[:, positive_label_idx][:, np.newaxis]
            for pt in prob_types:
                if best_C[pt] == C:
                    te_NLL = cal_metrics(decision_value_test, target_test, model_type, "alpha", (best_alpha if pt == "alpha" else 1.0))

                    res = results[pt]
                    res["dataset"].append(dn)
                    res["model_type"].append(model_type)
                    res["te_NLL"].append(te_NLL)
                    res["alpha"].append((best_alpha if pt == "alpha" else 1.0))
                    res["best_C"].append(C)
    for pt in prob_types:
        df = pd.DataFrame(results[pt])
        if search == "tuned":
            out_dir = f"tables/tune/{pt}"
        else:
            out_dir = f"tables/no_tune/{pt}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dn}.csv")
        df.to_csv(out_path, index=False)