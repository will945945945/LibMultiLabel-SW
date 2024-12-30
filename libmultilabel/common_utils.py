import copy
import json
import logging
import os
import time
from functools import wraps

import numpy as np
from scipy.special import log_expit, expit


class AttributeDict(dict):
    """AttributeDict is an extended dict that can access
    stored items as attributes.

    >>> ad = AttributeDict({'ans': 42})
    >>> ad.ans
    >>> 42
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_used", set())

    def __getattr__(self, key: str) -> any:
        try:
            value = self[key]
            self._used.add(key)
            return value
        except KeyError:
            raise AttributeError(f'Missing attribute "{key}"')

    def __setattr__(self, key: str, value: any) -> None:
        self[key] = value
        self._used.discard(key)

    def used_items(self) -> dict:
        """Returns the items that have been used at least once after being set.

        Returns:
            dict: the used items.
        """
        return {k: self[k] for k in self._used}


def dump_log(log_path, metrics=None, split=None, config=None):
    """Write log including the used items of config and the evaluation scores.

    Args:
        log_path(str): path to log path
        metrics (dict): metric and scores in dictionary format, defaults to None
        split (str): val or test, defaults to None
        config (dict): config to save, defaults to None
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.isfile(log_path):
        with open(log_path) as fp:
            result = json.load(fp)
    else:
        result = dict()

    if config:
        config_to_save = copy.deepcopy(config.used_items())
        config_to_save.pop("device", None)  # delete if device exists
        result["config"] = config_to_save
    if split and metrics:
        if split in result:
            result[split].append(metrics)
        else:
            result[split] = [metrics]

    with open(log_path, "w") as fp:
        json.dump(result, fp)

    logging.info(f"Finish writing log to {log_path}.")


def argsort_top_k(vals, k, axis=-1):
    """Get the indices of the top-k elements in a 2D array.

    Args:
        vals: Array to sort.
        k: Consider only the top k elements for each query
        axis: Axis along which to sort. The default is -1 (the last axis).

    Returns: Array of indices that sort vals along the specified axis.
    """
    unsorted_top_k_idx = np.argpartition(vals, -k, axis=axis)[:, -k:]
    unsorted_top_k_scores = np.take_along_axis(vals, unsorted_top_k_idx, axis=axis)
    sorted_order = np.argsort(-unsorted_top_k_scores, axis=axis)
    sorted_top_k_idx = np.take_along_axis(unsorted_top_k_idx, sorted_order, axis=axis)
    return sorted_top_k_idx


def is_multiclass_dataset(dataset, label="label"):
    """Determine whether the dataset is multi-class.

    Args:
        dataset (Union[list, scipy.sparse.csr_matrix]): The training dataset
            in `nn` or `linear` format.
        label (str, optional): Label key. Defaults to 'label'.

    Returns:
        bool: Whether the training dataset is mulit-class or not.
    """
    if isinstance(dataset, list):
        label_sizes = np.array([len(d[label]) for d in dataset])
    else:
        label_sizes = dataset[label].sum(axis=1)

    # TODO: separate logging message from the function
    # detect unlabeled ratio
    ratio = (label_sizes == 0).sum() / label_sizes.shape[0]
    threshold = 0.1
    if ratio >= threshold:
        logging.warning(
            f"""About {ratio * 100:.1f}% (>= {threshold * 100:.1f}%) instances in the dataset are unlabeled.
            LibMultiLabel doesn't treat unlabeled data in a special way.
            Thus, the metrics you see will not be accurate.
            We suggest you either apply preprocessing to the data or modify the metric classes."""
        )

    ratio = float((label_sizes == 1).sum()) / len(label_sizes)
    if ratio > 0.999 and ratio != 1.0:
        logging.info(
            f"""Only {(1-ratio)*100:.4f}% of training instances are multi-label.
            You may double check if your application should be a multi-label or
            a multi-class problem."""
        )
    return ratio == 1.0


def timer(func):
    """Log info-level wall time"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        value = func(*args, **kwargs)
        wall_time = time.time() - start_time
        logging.info(f"{repr(func.__name__)} finished in {wall_time:.2f} seconds")
        return value

    return wrapper

def l1_hinge_loss(x):
    '''return max(0, 1 - x)
    '''
    return np.maximum(0, 1 - x)

def l2_hinge_loss(x):
    '''return max(0, 1 - x)^2
    '''
    return np.maximum(0, 1 - x)**2

def decision_value_to_prob(decision_values, model_type, prob_type, use_log_prob, alpha=1.0, eps=1e-8):
    '''return probability corresponding to a specific model and probability transformation function
    Args:
        decision_values: decision values of a linear model ``wTx``
        model_type: type of linear models, ``l2svm``, ``l1svm``, and ``lr``
        prob_type: type of probability transformation functions, ``Exp`` and ``Prob``
        use_log_prob: If set to ``True``, return ``log(prob)``
        alpha: the corresponding parameter in the ``Prob`` probability transformation function ``sigmoid(-0.5*alpah*(loss(wTx) - loss(-wTx)))``
            Default: 1.0
        eps: a scalar close to zero, which is used to avoid numerical issues when calculating cross entropy
            Default: 1e-8
    '''
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
            return log_expit(decision_values)
        else:
            # sigmoid(wTx) for LR
            return expit(decision_values)
