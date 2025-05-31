import sys
import os

parent_dir = os.path.abspath("..")
sys.path.append(parent_dir)

import libmultilabel.linear as linear
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from tqdm import tqdm
from libmultilabel.common_utils import AttributeDict
from scipy.special import expit
from util import sigmoid_train, sigmoid_predict, gen_S


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
        if prob_type == "platt":
            eps = np.finfo(decision_values.dtype).eps
            prob = np.expand_dims(
                np.array([sigmoid_predict(float(x), A, B) for x in decision_values]), axis=-1
            )
            return np.where(prob == 1, 1.0 - eps, prob)


def cal_metrics(preds, target, model_type, prob_type=None, alpha=None, A=None, B=None):
    metrics_ce = linear.get_metrics(["CrossEntropy"], 2)
    # metrics_acc = linear.get_metrics(["P@1"], 2)
    
    probs = decision_value_to_prob(preds, prob_type, model_type, alpha, A, B)
    # CrossEntropy
    metrics_ce.update(probs, target)
    # Acc
    # probs = np.concatenate([1 - probs, probs], axis=1)
    # target = np.concatenate([1 - target, target], axis=1)
    # metrics_acc.update(probs, target)

    metrics_ce = metrics_ce.compute()
    # metrics_acc = metrics_acc.compute()

    return metrics_ce["CrossEntropy"]

def find_alpha_A_B(model_type, prob_type, train_data, test_data, param, positive_label_idx):
    X_train, y_train = train_data
    X_test, y_test = test_data
    alpha, A, B = None, None, None
    # Train the model
    model = linear.train_binary_and_multiclass(y_train, X_train, False, param)
    # Test Data
    decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
    target_test = y_test.toarray()[:, positive_label_idx][:, np.newaxis]

    if model_type != "lr":
        # Train Data (S)
        decision_value_train = gen_S(X_train, y_train, positive_label_idx, param)[:, np.newaxis]
        target_train = y_train.toarray()[:, positive_label_idx][:, np.newaxis]

        if prob_type == "alpha":
            # Grid search Alpha value
            _min = float("inf")
            best_alpha = 0
            # Use Whole Training Set
            # ======================
            for tmp_alpha in np.arange(1, 10.1, 0.1):
                metric = cal_metrics(decision_value_train, target_train, model_type, prob_type=prob_type, alpha=tmp_alpha
                )
                if metric < _min:
                    _min = metric
                    best_alpha = tmp_alpha

            alpha = best_alpha

        if prob_type == "platt":
            # Train Platt model.
            A, B = sigmoid_train(decision_value_train, target_train)

        if prob_type == "franc":
            alpha = 1
    
    ce = cal_metrics(
        decision_value_test, target_test, model_type, prob_type=prob_type, alpha=alpha, A=A, B=B
    )
    return ce, (alpha, A, B)

data_names = ["real-sim" ,"rcv1", "a9a", "ijcnn1", "webspam",]
prob_types = ["platt"]
model_types = ["lr", "l2svm", "l1svm", ]
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

for prob_type in prob_types:
    for dn in data_names:
        df = {_c: [] for _c in df_cols}
        for model_type in model_types:
            _min_ce = float('inf')
            best_C = 0
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
            for i in pbar_dn:
                C = 2 ** i 
                param = f"-s {model2s[model_type]} -c {C}"                
                cur_ce = 0
                kf = StratifiedKFold(n_splits=5, shuffle=False)
                for train_idx, test_idx in kf.split(X.toarray(), y.toarray()[:, positive_label_idx]):
                    train_data = (X[train_idx], y[train_idx])
                    test_data = (X[test_idx], y[test_idx])
                    
                    cur_ce += find_alpha_A_B(model_type, prob_type, train_data, test_data, param, positive_label_idx)[0]
                cur_ce /= 5
                if cur_ce < _min_ce:
                    _min_ce = cur_ce
                    best_C = C

            param = f"-s {model2s[model_type]} -c {best_C}" 
            te_NLL, (alpha, A, B) = find_alpha_A_B(model_type, prob_type, (X, y), (X_test, y_test), param, positive_label_idx)

            for col in df_cols:
                df[col].append(eval(col) if col != "dataset" else eval("dn"))

        df = pd.DataFrame(df)

        if search == "tuned":
            os.makedirs(f"tables/tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/tune/{prob_type}/{dn}.csv", index=False)
        else:
            os.makedirs(f"tables/no_tune/{prob_type}", exist_ok=True)
            df.to_csv(f"tables/no_tune/{prob_type}/{dn}.csv", index=False)
