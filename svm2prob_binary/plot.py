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
import matplotlib.pyplot as plt


def l1_hinge_loss(x):
    """return max(0, 1 - x)"""
    return np.maximum(0, 1 - x)


def l2_hinge_loss(x):
    """return max(0, 1 - x)^2"""
    return np.maximum(0, 1 - x) ** 2

def get_decision_value(model_type, X_train, y_train, X_test, param, positive_label_idx):
    # Train the model
    model = linear.train_binary_and_multiclass(y_train, X_train, False, param)
    # Train Data
    decision_value_train = model.predict_values(X_train)[:, positive_label_idx][:, np.newaxis]
    decision_value_test = model.predict_values(X_test)[:, positive_label_idx][:, np.newaxis]
    
    return decision_value_train, decision_value_test

def plot(data, path):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, alpha=0.6, label='Decision Value', color='blue', density=True)

    # Decorations
    plt.title("Histogram of Decision Values")
    plt.xlabel("Decision Value")
    plt.ylabel("Density")
    plt.legend()

    # Show the plot
    plt.savefig(path)
data_names = ["real-sim" ,"rcv1", "a9a", "ijcnn1", "webspam",]
# "cod-rna", "covtype", "mushrooms"
prob_types = ["platt", "alpha", "franc"]
model_types = ["lr", "l2svm", "l1svm", ]

import sys
model2s = {
    "l2svm":1,
    "l1svm":3,
    "lr":0
}
pbar = tqdm(data_names)
for dn in pbar:
    for tune in ["no_tune", "tune"]:
        for prob_type in prob_types:
            for model_type in model_types:
                if model_type == "lr" and prob_type != "platt":
                    continue
                ARGS = {
                        "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/trva.svm",
                        "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
                        "table": f"tables_v1/{tune}/{prob_type}/{dn}.csv"
                    }
                ARGS = AttributeDict(ARGS)
                # Load Data
                datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
                preprocessor = linear.Preprocessor(False, False)
                datasets = preprocessor.fit_transform(datasets)
                positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]
                X_train, X_test = datasets["train"]["x"], datasets["test"]["x"]
                y_train = datasets["train"]["y"]
                print(X_train.shape, X_test.shape)
                breakpoint()
                # Load Best C
                df = pd.read_csv(ARGS.table)
                C = df.loc[(df['dataset'] == dn) & (df['model_type'] == model_type), 'best_C'].values[0]
                param = f"-s {model2s[model_type]} -c {C}"                
                d_train, d_test = get_decision_value(model_type, X_train, y_train, X_test, param, positive_label_idx)
                os.makedirs(f"plot/{tune}/{prob_type}/{dn}", exist_ok=True)
                plot(d_train, f"plot/{tune}/{prob_type}/{dn}/{model_type}_train.png")
                plot(d_test, f"plot/{tune}/{prob_type}/{dn}/{model_type}.png")