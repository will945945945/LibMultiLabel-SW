from __future__ import annotations

from typing import Callable

import numpy as np
import scipy.sparse as sparse
import sklearn.cluster
import sklearn.preprocessing
from tqdm import tqdm
import psutil

from . import linear

__all__ = [
        "train_tree", 
        "train_random_partitions",
        "train_random_label_forests_with_partitions", 
        "train_tree_subsample",
        "train_random_selection", 
        #"train_random_label_forests_100U"
        ]


class Node:
    def __init__(
        self,
        label_map: np.ndarray,
        children: list[Node],
        is_root=None
    ):
        """
        Args:
            label_map (np.ndarray): The labels under this node.
            children (list[Node]): Children of this node. Must be an empty list if this is a leaf node.
        """
        self.label_map = label_map
        self.children = children
        self.is_root = is_root

    def isLeaf(self) -> bool:
        return len(self.children) == 0

    def dfs(self, visit: Callable[[Node], None]):
        visit(self)
        # Stops if self.children is empty, i.e. self is a leaf node
        for child in self.children:
            child.dfs(visit)


class TreeModel:
    """A model returned from train_tree."""

    def __init__(
        self,
        root: Node,
        flat_model: linear.FlatModel,
        weight_map: np.ndarray,
    ):
        self.name = "tree"
        self.root = root
        self.flat_model = flat_model
        self.weight_map = weight_map
        self.multiclass = False

    def predict_values(
        self,
        x: sparse.csr_matrix,
        beam_width: int = 10,
    ) -> np.ndarray:
        #level_0_model: linear.FlatModel,
        """Calculates the decision values associated with x.

        Args:
            x (sparse.csr_matrix): A matrix with dimension number of instances * number of features.
            beam_width (int, optional): Number of candidates considered during beam search. Defaults to 10.

        Returns:
            np.ndarray: A matrix with dimension number of instances * number of classes.
        """
        # level_0_pred 
        # level_0_pred = linear.predict_values(level_0_model, x)

        # number of instances * number of labels + total number of metalabels
        all_preds = linear.predict_values(self.flat_model, x)
        #return np.vstack([self._beam_search(all_preds[i], beam_width) for i in range(all_preds.shape[0])])

        #return sparse.vstack([ sparse.csr_matrix( self._beam_search(level_0_pred[i], all_preds[i], beam_width) ) for i in range(all_preds.shape[0])])
        return sparse.vstack([ sparse.csr_matrix( self._beam_search(all_preds[i], beam_width) ) for i in range(all_preds.shape[0])])

    #def _beam_search(self, level_0_pred: np.ndarray, instance_preds: np.ndarray, beam_width: int) -> np.ndarray:
    def _beam_search(self, instance_preds: np.ndarray, beam_width: int) -> np.ndarray:
        """Predict with beam search using cached decision values for a single instance.

        Args:
            instance_preds (np.ndarray): A vector of cached decision values of each node, has dimension number of labels + total number of metalabels.
            beam_width (int): Number of candidates considered.

        Returns:
            np.ndarray: A vector with dimension number of classes.
        """
        cur_level = [(self.root, 0.0)]  # pairs of (node, score)
        #cur_level = [(self.root, -np.maximum(0, 1 - level_0_pred) ** 2)]  # pairs of (node, score)
        next_level = []
        while True:
            num_internal = sum(map(lambda pair: not pair[0].isLeaf(), cur_level))
            if num_internal == 0:
                break

            for node, score in cur_level:
                if node.isLeaf():
                    next_level.append((node, score))
                    continue
                slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
                pred = instance_preds[slice]
                children_score = score - np.maximum(0, 1 - pred) ** 2
                next_level.extend(zip(node.children, children_score.tolist()))

            cur_level = sorted(next_level, key=lambda pair: -pair[1])[:beam_width]
            next_level = []

        num_labels = len(self.root.label_map)
        #scores = np.full(num_labels, -np.inf)
        scores = np.full(num_labels, 0.0)
        for node, score in cur_level:
            slice = np.s_[self.weight_map[node.index] : self.weight_map[node.index + 1]]
            pred = instance_preds[slice]
            scores[node.label_map] = np.exp(score - np.maximum(0, 1 - pred) ** 2)
        return scores

def train_random_selection(
    y: sparse.csr_matrix, x: sparse.csr_matrix, options: str = "", K=100, dmax=10, sample_rate=0.1, verbose: bool = True,) -> TreeModel:
    """Random Selections in RLF paper.
    """
    def subsample_indices(y, sample_rate):
        indices = np.random.choice(y.shape[1], int(y.shape[1]*sample_rate), replace=False, p=np.ones(y.shape[1])/y.shape[1] )
        indices = np.sort(indices)
        return indices.tolist()

    indices = subsample_indices(y, sample_rate)
    y_level_1 = y[:,indices]

    # level 0's binary
    y_level_0 = np.sum(y_level_1, axis=1) > 1
    y_level_0 = y_level_0.astype(int)
    y_level_0 = sparse.csr_matrix(y_level_0)
    level_0_model = linear.train_1vsrest(y_level_0, x, False, options, verbose)

    # level 1's tree-based 
    label_representation = (y_level_1.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree(label_representation, np.arange(y_level_1.shape[1]), 0, K, dmax)
    root.is_root = True

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        # binary in head
        relevant_instances = y_level_1[:, node.label_map].getnnz(axis=1) > 0

        _train_node(y_level_1[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return level_0_model, TreeModel(root, flat_model, weight_map), indices


def train_tree_subsample(
    y: sparse.csr_matrix, x: sparse.csr_matrix, options: str = "", K=100, dmax=10, sample_rate=0.1, verbose: bool = True,) -> TreeModel:
    """Random Selections
    """
    # def subsample_indices(num_label: int, sample_rate: float) -> list:
    #     indices = []
    #     for idx in range(num_label):
    #         if np.random.uniform(low=0.0, high=1.0) < sample_rate:
    #             indices += [idx]
    #     return indices
    def subsample_indices(y, sample_rate):
        # # label dist
        # total_labels = np.sum(y) 
        # label_dist = np.sum(y, axis=0)/total_labels
        # label_dist = np.squeeze( np.asarray(label_dist) )
        # indices = np.random.choice(y.shape[1], int(y.shape[1]*sample_rate), replace=False, p=label_dist )

        # uniform dist
        indices = np.random.choice(y.shape[1], int(y.shape[1]*sample_rate), replace=False, p=np.ones(y.shape[1])/y.shape[1] )
        indices = np.sort(indices)
        return indices.tolist()

    #indices = subsample_indices(y.shape[1], sample_rate)
    indices = subsample_indices(y, sample_rate)
    y_level_1 = y[:,indices]

    # level 0's binary
    y_level_0 = np.sum(y_level_1, axis=1) > 1
    y_level_0 = y_level_0.astype(int)
    y_level_0 = sparse.csr_matrix(y_level_0)
    level_0_model = linear.train_1vsrest(y_level_0, x, False, options, verbose)

    # level 1's tree-based 
    label_representation = (y_level_1.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree(label_representation, np.arange(y_level_1.shape[1]), 0, K, dmax)
    root.is_root = True

    num_nodes = 0

    def count(node):
        nonlocal num_nodes
        num_nodes += 1

    root.dfs(count)

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        # # no binary in head
        # if node.is_root == False:
        #     relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        # else:
        #     relevant_instances = y[:, node.label_map].getnnz(axis=1) >= 0

        # binary in head
        relevant_instances = y_level_1[:, node.label_map].getnnz(axis=1) > 0

        _train_node(y_level_1[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return level_0_model, TreeModel(root, flat_model, weight_map), indices


def train_tree(
    y: sparse.csr_matrix, x: sparse.csr_matrix, options: str = "", K=100, dmax=10, verbose: bool = True,) -> TreeModel:
    """Trains a linear model for multi-label data using a divide-and-conquer strategy.
    The algorithm used is based on https://github.com/xmc-aalto/bonsai.
    """
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree(label_representation, np.arange(y.shape[1]), 0, K, dmax)

    num_nodes = 0
    features_used_perlabel = (x != 0).T * y

    def count(node):
        nonlocal num_nodes
        num_nodes += 1
        node.num_features_used = np.count_nonzero(features_used_perlabel[:, node.label_map].sum(axis=1))

    root.dfs(count)

    model_size = get_estimated_model_size(root)
    print(f'The estimated tree model size is: {model_size / (1024**3):.3f} GB')

    total_memory = psutil.virtual_memory().total 
    print(f'Your system memory is: {total_memory / (1024**3):.3f} GB')

    if (total_memory <= model_size):
        raise MemoryError(f'Not enough memory to train the model.')

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        if node.is_root == False:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        else:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) >= 0
        _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)
    return TreeModel(root, flat_model, weight_map), root.is_root

def partition_labels(num_labels, K):
    metalabels = []
    filter_pool = []
    counter = np.zeros(K, dtype=int)
    while len(metalabels) < num_labels:
        label_partition = np.random.choice(K)
        if label_partition not in filter_pool:
            counter[label_partition] += 1
            metalabels += [label_partition]
            if counter[label_partition] == int( num_labels / K ):
                filter_pool += [label_partition]

        if len(filter_pool) == K:
            break

    diff = num_labels - len(metalabels)
    if diff > 1:
        metalabels += [i for i in np.random.choice(K, size=K, replace=False)[:diff] ]
    elif diff == 1:
        metalabels += [np.random.choice(K)]
    metalabels = np.array(metalabels)

    return metalabels

def _build_tree_with_partitions(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Builds the tree recursively by kmeans clustering, but uses random partition in first level."""
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    if d == 0:
        metalabels = partition_labels(label_representation.shape[0], K)
        children = []
        for i in range(K):
            child_representation = label_representation[metalabels == i]
            child_map = label_map[metalabels == i]
            # we still set #clusters = 100 when using K-means.
            child = _build_tree(child_representation, child_map, d + 1, 100, dmax)
            children.append(child)
        return Node(label_map=label_map, children=children, is_root=metalabels)

    else:
        # we still set #clusters = 100 when using K-means.
        metalabels = (
            sklearn.cluster.KMeans(
                100, random_state=np.random.randint(2**31 - 1), n_init=1, max_iter=300, tol=0.0001, algorithm="elkan")
            .fit(label_representation)
            .labels_
        )
        children = []
        for i in range(100):
            child_representation = label_representation[metalabels == i]
            child_map = label_map[metalabels == i]
            child = _build_tree(child_representation, child_map, d + 1, 100, dmax)
            children.append(child)
        return Node(label_map=label_map, children=children)

def train_random_partitions(
        y: sparse.csr_matrix, x: sparse.csr_matrix, options: str = "", K=100, dmax=10, verbose: bool = True,) -> TreeModel:
    """Random partitions model in RLF paper."""
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)
    root = _build_tree_with_partitions(label_representation, np.arange(y.shape[1]), 0, K, dmax)

    num_nodes = 0
    features_used_perlabel = (x != 0).T * y

    def count(node):
        nonlocal num_nodes
        num_nodes += 1
        node.num_features_used = np.count_nonzero(features_used_perlabel[:, node.label_map].sum(axis=1))

    root.dfs(count)

    model_size = get_estimated_model_size(root)
    print(f'The estimated tree model size is: {model_size / (1024**3):.3f} GB')

    total_memory = psutil.virtual_memory().total 
    print(f'Your system memory is: {total_memory / (1024**3):.3f} GB')

    if (total_memory <= model_size):
        raise MemoryError(f'Not enough memory to train the model.')

    pbar = tqdm(total=num_nodes, disable=not verbose)

    def visit(node):
        if node.is_root == False:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) > 0
        else:
            relevant_instances = y[:, node.label_map].getnnz(axis=1) >= 0
        _train_node(y[relevant_instances], x[relevant_instances], options, node)
        pbar.update()

    root.dfs(visit)
    pbar.close()

    flat_model, weight_map = _flatten_model(root)

    return TreeModel(root, flat_model, weight_map), root.is_root

def train_random_label_forests_with_partitions(
    y: sparse.csr_matrix, x: sparse.csr_matrix, options: str = "", K=10, dmax=10, verbose: bool = True,) -> TreeModel:
    """Random label forests with partitions on labels in RLF paper."""
    label_representation = (y.T * x).tocsr()
    label_representation = sklearn.preprocessing.normalize(label_representation, norm="l2", axis=1)

    metalabels = partition_labels(label_representation.shape[0], K)

    models = []
    for i in range(K):
        # we still set #clusters = 100 when using K-means.
        models += [ train_tree( y[:, metalabels==i], x, options, 100, dmax) ]

    return models, metalabels


def _build_tree(label_representation: sparse.csr_matrix, label_map: np.ndarray, d: int, K: int, dmax: int) -> Node:
    """Builds the tree recursively by kmeans clustering."""
    if d >= dmax or label_representation.shape[0] <= K:
        return Node(label_map=label_map, children=[])

    metalabels = (
        sklearn.cluster.KMeans(
            K, random_state=np.random.randint(2**31 - 1), n_init=1, max_iter=300, tol=0.0001, algorithm="elkan")
        .fit(label_representation)
        .labels_
    )
    children = []
    for i in range(K):
        child_representation = label_representation[metalabels == i]
        child_map = label_map[metalabels == i]
        child = _build_tree(child_representation, child_map, d + 1, K, dmax)
        children.append(child)
    return Node(label_map=label_map, children=children)


def get_estimated_model_size(root):
    total_num_weights = 0

    def collect_stat(node: Node):
        nonlocal total_num_weights
        
        if node.isLeaf():
            total_num_weights += len(node.label_map) * node.num_features_used
        else:
            total_num_weights += len(node.children) * node.num_features_used

    root.dfs(collect_stat)

    # 16 is because when storing sparse matrices, indices (int64) require 8 bytes and floats require 8 bytes
    # Our study showed that among the used features of every binary classification problem, on average no more than 2/3 of weights obtained by the dual coordinate descent method are non-zeros.
    return total_num_weights * 16 * 2/3


def _train_node(y: sparse.csr_matrix, x: sparse.csr_matrix, options: str, node: Node):
    """If node is internal, computes the metalabels representing each child and trains
    on the metalabels. Otherwise, train on y.

    Args:
        y (sparse.csr_matrix): A 0/1 matrix with dimensions number of instances * number of classes.
        x (sparse.csr_matrix): A matrix with dimensions number of instances * number of features.
        options (str): The option string passed to liblinear.
        node (Node): Node to be trained.
    """
    if node.isLeaf():
        node.model = linear.train_1vsrest(y[:, node.label_map], x, False, options, False)
    else:
        # meta_y[i, j] is 1 if the ith instance is relevant to the jth child.
        # getnnz returns an ndarray of shape number of instances.
        # This must be reshaped into number of instances * 1 to be interpreted as a column.
        meta_y = [y[:, child.label_map].getnnz(axis=1)[:, np.newaxis] > 0 for child in node.children]
        meta_y = sparse.csr_matrix(np.hstack(meta_y))
        node.model = linear.train_1vsrest(meta_y, x, False, options, False)

    node.model.weights = sparse.csc_matrix(node.model.weights)


def _flatten_model(root: Node) -> tuple[linear.FlatModel, np.ndarray]:
    """Flattens tree weight matrices into a single weight matrix. The flattened weight
    matrix is used to predict all possible values, which is cached for beam search.
    This pessimizes complexity but is faster in practice.
    Consecutive values of the returned map denotes the start and end indices of the
    weights of each node. Conceptually, given root and node:
        flat_model, weight_map = _flatten_model(root)
        slice = np.s_[weight_map[node.index]:
                      weight_map[node.index+1]]
        node.model.weights == flat_model.weights[:, slice]

    Args:
        root (Node): Root of the tree.

    Returns:
        tuple[linear.FlatModel, np.ndarray]: The flattened model and the ranges of each node.
    """
    index = 0
    weights = []
    bias = root.model.bias

    def visit(node):
        assert bias == node.model.bias
        nonlocal index
        node.index = index
        index += 1
        weights.append(node.model.__dict__.pop("weights"))

    root.dfs(visit)

    model = linear.FlatModel(
        name="flattened-tree",
        weights=sparse.hstack(weights, "csc"),
        bias=bias,
        thresholds=0,
        multiclass=False,
    )
    #weights=sparse.hstack(weights, "csr"), # memory issue

    # w.shape[1] is the number of labels/metalabels of each node
    weight_map = np.cumsum([0] + list(map(lambda w: w.shape[1], weights)))

    return model, weight_map
