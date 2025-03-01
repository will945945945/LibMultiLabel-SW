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


# In[2]:


def l1_hinge_loss(x):
    '''return max(0, 1 - x)
    '''
    return np.maximum(0, 1 - x)

def l2_hinge_loss(x):
    '''return max(0, 1 - x)^2
    '''
    return np.maximum(0, 1 - x)**2

def decision_value_to_prob(decision_values, model_type, prob_type, use_log_prob, alpha=1.0):
    '''return probability corresponding to a specific model and probability transformation function
    Args:
        decision_values: decision values of a linear model ``wTx``
        model_type: type of linear models, ``l2svm``, ``l1svm``, and ``lr``
        prob_type: type of probability transformation functions, ``Exp`` and ``Prob``
        use_log_prob: If set to ``True``, return ``log(prob)``
        alpha: the corresponding parameter in the ``Prob`` probability transformation function ``sigmoid(-0.5*alpah*(loss(wTx) - loss(-wTx)))``
            Default: 1.0
    '''
    #eps: a scalar close to zero, which is used to avoid numerical issues when calculating cross entropy
    eps = np.finfo(decision_values.dtype).eps

    model_type = model_type.lower()
    prob_type = prob_type.lower()
    assert model_type in ["l2svm", "l1svm", "lr"], "Our experiments only cover three kinds of models: l2-SVM, l1-SVM, and LR."
    assert prob_type in ["exp", "prob"], "There are only two kinds of probability transformation functions: Exp and Prob."
    assert not (model_type == "lr" and prob_type == "prob"), "Logits from Logistic Regression only support Exp."

    loss_func = l2_hinge_loss if model_type == "l2svm" else l1_hinge_loss
    if model_type != "lr":
        if prob_type == "prob":
            if use_log_prob:
                # log_sigmoid(-0.5*alpah*(loss(wTx) - loss(-wTx))) for l1/l2-SVM when using Prob
                return log_expit(-0.5 * alpha * (loss_func(decision_values) - loss_func(-decision_values)))
            else:
                # sigmoid(-0.5*alpah*(loss(wTx) - loss(-wTx))) for l1/l2-SVM when using Prob
                prob = expit(-0.5 * alpha * (loss_func(decision_values) - loss_func(-decision_values)))
                return np.where(prob == 1, # condition
                                1.0 - eps, # for wTx >= 1.0, add eps to avoid numerical issues when calculating cross entropy
                                prob
                               )
        else:
            if use_log_prob:
                # -loss(wTx) for l1/l2-SVM when using Exp
                return -loss_func(decision_values)
            else:
                # exp(-loss(wTx)) for l1/l2-SVM when using Exp
                return np.where(decision_values >= 1, # condition
                                1.0 - eps, # for wTx >= 1.0, add eps to avoid numerical issues when calculating cross entropy
                                np.exp(-loss_func(decision_values))
                               )
    else:
        if use_log_prob:
            # log_sigmoid(wTx) for LR
            return log_expit(alpha*decision_values)
        else:
            # sigmoid(wTx) for LR
            prob = expit(alpha*decision_values)
            return np.where(prob == 1, # condition
                            1.0 - eps, # for wTx >= 1.0, add eps to avoid numerical issues when calculating cross entropy
                            prob
                           )


# In[3]:


def metrics_in_batches(model, batch_size, datasets, model_type, prob_types, positive_label_idx):
    num_instances = datasets["x"].shape[0]
    num_batches = math.ceil(num_instances / batch_size)
    
    metrics = {_pt:linear.get_metrics(["CrossEntropy"], datasets["y"].shape[1]) for _pt in prob_types}
    # for i in tqdm(range(num_batches)):
    for i in range(num_batches):
        tmp_data = datasets["x"][i * batch_size : (i + 1) * batch_size]
        preds = model.predict_values(tmp_data)[:, positive_label_idx][:, np.newaxis]
        target = datasets["y"][i * batch_size : (i + 1) * batch_size].toarray()[:, positive_label_idx][:, np.newaxis]
        for pt in prob_types:
            prob_type, alpha = pt.split("-")
            probs = decision_value_to_prob(preds, model_type, prob_type, use_log_prob=False, alpha=float(alpha))
            # print(pt, probs.shape, f"{probs.min():.4e}", f"{probs.max():.4e}")
            metrics[pt].update(probs, target)
    for pt in prob_types:
        metrics[pt] = metrics[pt].compute()
    # print(metrics)
    return metrics


# In[4]:


prob_types = list(f"prob-{_a}" for _a in np.arange(1, 4, 0.5))
prob_alpha_num = len(prob_types)
prob_types.extend(list(f"exp-{_a}" for _a in np.arange(1, 4, 0.5)))
data_names = ["a9a", "ijcnn1", "webspam", "real-sim", "rcv1"]
model_types = ["l2svm", "l1svm", "lr"]
modes = ["trvate", "trva"]
df_cols = "dataset,mode,model_type,prob_type,tr_NLL,te_NLL".split(",")

for dn in tqdm(data_names):
    df = {_c:[] for _c in df_cols}
    for model_type in model_types:
        for mode in modes:
            # print(mode)
            logs_dir = f"../runs/{mode}"
            if model_type == "lr":
                c = 10
                _prob_types = prob_types[prob_alpha_num:]
            else:
                c = 1
                _prob_types = prob_types[:prob_alpha_num+1]
            model_path_prefix = f"{dn}_{model_type}_c{c}"
            model_path = sorted([os.path.join(logs_dir, _d) for _d in os.listdir(logs_dir) if _d.startswith(model_path_prefix)])[-1] # take the last one as it should be the lastest one
            ARGS = {
                "traindata_path": f"../datasets/binary_datasets/dataset_{dn}/{mode}.svm",
                "testdata_path": f"../datasets/binary_datasets/dataset_{dn}/te.svm",
                "modelpath": f"{model_path}/linear_pipeline.pickle"
            }
            ARGS = AttributeDict(ARGS)
            
            datasets = linear.load_dataset("svm", ARGS.traindata_path, ARGS.testdata_path)
            preprocessor = linear.Preprocessor(False, False)
            datasets = preprocessor.fit_transform(datasets)
            positive_label_idx = np.where(preprocessor.label_mapping == 1)[0][0]
            
            with open(ARGS.modelpath, "rb") as F:
                model = pickle.load(F)['model']
                
            tr_metrics = metrics_in_batches(model, 2**16, datasets["train"], model_type, _prob_types, positive_label_idx)
            te_metrics = metrics_in_batches(model, 2**16, datasets["test"], model_type, _prob_types, positive_label_idx)
            for prob_type in _prob_types:
                tr_NLL = tr_metrics[prob_type]['CrossEntropy']
                # tr_Acc = tr_metrics[prob_type]['P@1']
                te_NLL = te_metrics[prob_type]['CrossEntropy']
                # te_Acc = te_metrics[prob_type]['P@1']
                for col in df_cols:
                    df[col].append(eval(col) if col != "dataset" else eval("dn"))
    df = pd.DataFrame(df)
    df.to_csv(f"{dn}_res.csv", index=False)


# In[7]:


def plot_grouped_lines(df, mode, entropy, ax):
    # Filter the DataFrame based on the mode
    data_name = df["dataset"].iloc[0]
    filtered_df = df[df['mode'] == mode]
    
    # Get unique model types in the filtered DataFrame
    model_types = ["l1svm", "l2svm", "lr"]
    colors = ["b", "y", "g"]
    
    # Plot for each model type
    for i, model_type in enumerate(model_types):
        model_df = filtered_df[filtered_df['model_type'] == model_type]
        # print(model_df.to_string(index=False))

        if model_type != "lr":
            marker, prob_type = "o", "Prob"
            x = model_df['prob_type'].apply(lambda x: float(x.split("-")[-1]))[:-1]
            y_tr = model_df['tr_NLL'][:-1]
            y_te = model_df['te_NLL'][:-1]
        else:
            marker, prob_type = "*", "Exp"
            x = model_df['prob_type'].apply(lambda x: float(x.split("-")[-1]))[:]
            y_tr = model_df['tr_NLL'][:]
            y_te = model_df['te_NLL'][:]
        # Extract x (from prob_type) and y (tr_NLL) values

        # Plot the line for tr_NLL vs prob-{n}
        ax.plot(x, y_tr, label=f'{model_type} - {prob_type} - NLL_on_{mode}', marker=marker, c=colors[i])
        ax.plot(x, y_te, label=f'{model_type} - {prob_type} - NLL_on_te', linestyle=':', marker=marker, color=colors[i])
        
        # Get y-value for horizontal line (where prob_type == "exp-1.0")
        if model_type != "lr":
            horizontal_line_row = model_df[model_df['prob_type'] == 'exp-1.0']
            if not horizontal_line_row.empty:
                y_horizontal_tr = horizontal_line_row['tr_NLL'].iloc[0]
                y_horizontal_te = horizontal_line_row['te_NLL'].iloc[0]
                
                # Plot the horizontal line
                ax.axhline(y=y_horizontal_tr, color=colors[i], label=f'{model_type} - Exp - NLL_on_{mode}')
                ax.axhline(y=y_horizontal_te, color=colors[i], linestyle=':', label=f'{model_type} - Exp - NLL_on_te')

    ax.axhline(y=entropy, color="r", linestyle='--', label=f'entropy_on_trvate')
    
    # Customize the plot
    # ax.set_ylim([0, 0.693])
    ax.set_xlabel('alpha (from prob-{alpha})')
    ax.set_ylabel('NLL')
    ax.set_title(f'{data_name}: NLL for models trained with {mode}')
    ax.legend()
    ax.grid(True)

## The positive ratios below are calculated on the corresponding datasets by
##   `awk '{if ($1 == "-1") neg++; total++} END {print "Ratio:", 1-neg/total}' trvate.svm`
pos_ratios = {
    "a9a": 0.23928176569346055,
    "ijcnn1": 0.09573649702521685,
    "webspam": 0.6062542857142857,
    "real-sim": 0.30754124659447646,
    "rcv1": 0.524554892846034,
}

for dn in data_names:
    df = pd.read_csv(f"{dn}_res.csv")
    pratio = pos_ratios[dn]
    entropy = -(pratio * np.log(pratio) + (1 - pratio) * np.log(1 - pratio))
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    plot_grouped_lines(df, 'trva', entropy, axs[0])
    plot_grouped_lines(df, 'trvate', entropy, axs[1])
    fig.savefig(f"{dn}.png", format="png", dpi=1000, bbox_inches="tight", transparent=True)
    #plt.show()


# In[ ]:




