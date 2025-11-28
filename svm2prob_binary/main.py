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
from util import sigmoid_train, sigmoid_predict, gen_S

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def l1_hinge_loss(x):
    """return max(0, 1 - x)"""
    return np.maximum(0, 1 - x)


def l2_hinge_loss(x):
    """return max(0, 1 - x)^2"""
    return np.maximum(0, 1 - x) ** 2


def decision_value_to_prob(decision_values, prob_type, model_type, alpha=None, A=None, B=None):
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
        if prob_type == "platt":
            prob = np.expand_dims(
                np.array([sigmoid_predict(float(x), A, B) for x in decision_values]),
                axis=-1,
            )
            return np.where(prob == 1, 1.0 - eps, prob)


def cal_metrics(preds, target, model_type, prob_type=None, alpha=None, A=None, B=None):
    metrics_ce = linear.get_metrics(["CrossEntropy"], 2)

    probs = decision_value_to_prob(preds, prob_type, model_type, alpha, A, B)
    metrics_ce.update(probs, target)
    metrics_ce = metrics_ce.compute()

    return metrics_ce["CrossEntropy"]


def find_all_ce(model_type, train_data, test_data, param, positive_label_idx):
    """
    Train ONE model with given (model_type, param) on train_data,
    then:

      - For non-LR models, use gen_S on train_data to tune alpha for the "alpha" method.
      - For Franc, fix alpha=1.
      - Optionally compute Platt (A,B).

    Finally, compute CE on test_data for all methods using the SAME decision values.

    Returns:
        {
          "alpha": (ce_alpha, alpha_best, None, None),
          "franc": (ce_franc, 1.0,      None, None),
          "platt": (ce_platt, None,     A,    B)   # ce_platt can be None if model_type == "lr"
        }
    """
    X_train, y_train = train_data
    X_test, y_test = test_data

    model = linear.train_binary_and_multiclass(y_train, X_train, False, param)

    # Decision values on test split
    decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
    target_test = y_test.toarray()[:, positive_label_idx][:, np.newaxis]

    alpha_best, A, B = None, None, None

    if model_type != "lr":
        decision_value_train = gen_S(X_train, y_train, positive_label_idx, param)[:, np.newaxis]
        target_train = y_train.toarray()[:, positive_label_idx][:, np.newaxis]

        _min = float("inf")
        best_alpha = 0.0
        for tmp_alpha in [i / 10 for i in range(10, 101)]:
            metric = cal_metrics(
                decision_value_train,
                target_train,
                model_type,
                prob_type="alpha",
                alpha=tmp_alpha,
            )
            if metric < _min:
                _min = metric
                best_alpha = tmp_alpha
        alpha_best = best_alpha

        A, B = sigmoid_train(decision_value_train, target_train)

    # alpha method (tuned alpha_best)
    ce_alpha = cal_metrics(
        decision_value_test,
        target_test,
        model_type,
        prob_type="alpha",
        alpha=alpha_best,
    )

    # Franc method (alpha fixed to 1)
    ce_franc = cal_metrics(
        decision_value_test,
        target_test,
        model_type,
        prob_type="franc",
        alpha=1.0,
    )

    # Platt method
    ce_platt = cal_metrics(
        decision_value_test,
        target_test,
        model_type,
        prob_type="platt",
        A=A,
        B=B,
    )
    if alpha_best == 1 and ce_alpha != ce_franc:
        print(ce_alpha, ce_franc)
    return {
        "alpha": (ce_alpha, alpha_best, None, None),
        "franc": (ce_franc, 1.0, None, None),
        "platt": (ce_platt, None, A, B),
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

# We are focusing on alpha and franc here
prob_types = ["alpha", "franc", "platt"]
model_types = ["lr", "l2svm", "l1svm"]
df_cols = "dataset,model_type,te_NLL,alpha,A,B,best_C".split(",")

search = sys.argv[1]
if search == "tuned":
    space = [i for i in range(-13, 11)]
else:
    space = [1]

model2s = {
    "l2svm": 1,
    "l1svm": 3,
    "lr": 0,
}

n_splits = 5

for dn in data_names:
    # Prepare per-method result dicts; we will write one CSV per prob_type
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

        pbar_dn = tqdm(space)
        pbar_dn.set_description(f"Dataset: {dn}, Model: {model_type}")

        for i in pbar_dn:
            C = 2 ** i
            param = f"-s {model2s[model_type]} -c {C}"

            # Accumulate CE over folds for all methods
            cur_ce = {pt: 0.0 for pt in prob_types}
            kf = StratifiedKFold(n_splits=n_splits, shuffle=False)

            for train_idx, test_idx in kf.split(X.toarray(), y.toarray()[:, positive_label_idx]):
                train_data = (X[train_idx], y[train_idx])
                test_data = (X[test_idx], y[test_idx])

                ce_dict = find_all_ce(
                    model_type,
                    train_data,
                    test_data,
                    param,
                    positive_label_idx,
                )

                for pt in prob_types:
                    cur_ce[pt] += ce_dict[pt][0]

            for pt in prob_types:
                cur_ce[pt] /= n_splits
                if cur_ce[pt] < best_ce[pt]:
                    best_ce[pt] = cur_ce[pt]
                    best_C[pt] = C

        # print(best_C, best_ce)
        unique_Cs = sorted(set(best_C.values()))
        full_eval = {}

        for C in unique_Cs:
            param = f"-s {model2s[model_type]} -c {C}"
            ce_dict = find_all_ce(
                model_type,
                (X, y),
                (X_test, y_test),
                param,
                positive_label_idx,
            )
            full_eval[C] = ce_dict

        for pt in prob_types:
            C_star = best_C[pt]
            te_NLL, alpha, A, B = full_eval[C_star][pt]

            res = results[pt]
            res["dataset"].append(dn)
            res["model_type"].append(model_type)
            res["te_NLL"].append(te_NLL)
            res["alpha"].append(alpha)
            res["A"].append(A)
            res["B"].append(B)
            res["best_C"].append(C_star)

    for pt in prob_types:
        df = pd.DataFrame(results[pt])
        if search == "tuned":
            out_dir = f"tables/tune/{pt}"
        else:
            out_dir = f"tables/no_tune/{pt}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{dn}.csv")
        df.to_csv(out_path, index=False)
