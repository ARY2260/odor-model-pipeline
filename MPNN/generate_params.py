import numpy as np
import os
import tempfile
import collections
import logging
import itertools
from typing import Dict, List, Optional, Tuple, Any, Callable

from deepchem.data import Dataset
from deepchem.trans import Transformer
from deepchem.models import Model
from deepchem.metrics import Metric
from deepchem.hyper.base_classes import HyperparamOpt
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename

logger = logging.getLogger(__name__)


n_trials = 500

params_dict = {"batch_size": [],
                # "n_tasks": ,
                "class_imbalance_ratio": ,
                "node_out_feats": ,
                "edge_hidden_feats": ,
                "num_step_message_passing": ,
                # "mode": 'classification',
                "number_atom_features": ,
                "number_bond_features": ,
                # "n_classes": ,
                "ffn_hidden_list": [[]],
                # "ffn_embeddings": ,
                "ffn_activation": 'relu',
                "ffn_dropout_p": 0.0,
                "ffn_dropout_at_input_no_act": True,
                "weight_decay": ,
                "self_loop": ,
                "learning_rate":
                }

def generate_random_hyperparam_values(params_dict: Dict,
                                          n: int) -> List[Dict[str, Any]]:
    """Generates `n` random hyperparameter combinations of hyperparameter values
    Parameters
    ----------
    params_dict: Dict
        A dictionary of hyperparameters where parameter which takes discrete
        values are specified as iterables and continuous parameters are of
        type callables.
    n: int
        Number of hyperparameter combinations to generate
    Returns
    -------
    A list of generated hyperparameters
    Example
    -------
    >>> from scipy.stats import uniform
    >>> from deepchem.hyper import RandomHyperparamOpt
    >>> n = 1
    >>> params_dict = {'a': [1, 2, 3], 'b': [5, 7, 8], 'c': uniform(10, 5).rvs}
    >>> RandomHyperparamOpt.generate_random_hyperparam_values(params_dict, n)  # doctest: +SKIP
    [{'a': 3, 'b': 7, 'c': 10.619700740985433}]
    """
    hyperparam_keys, hyperparam_values = [], []
    for key, values in params_dict.items():
        if callable(values):
            # If callable, sample it for a maximum n times
            values = [values() for i in range(n)]
        hyperparam_keys.append(key)
        hyperparam_values.append(values)
    hyperparam_combs = []
    for iterable_hyperparam_comb in itertools.product(*hyperparam_values):
        hyperparam_comb = list(iterable_hyperparam_comb)
        hyperparam_combs.append(hyperparam_comb)
    indices = np.random.permutation(len(hyperparam_combs))[:n]
    params_subset = []
    for index in indices:
        param = {}
        for key, hyperparam_value in zip(hyperparam_keys,
                                         hyperparam_combs[index]):
            param[key] = hyperparam_value
        params_subset.append(param)
    return params_subset

hyperparameter_combs = generate_random_hyperparam_values(params_dict=params_dict, n=n_trials)

_convert_hyperparam_dict_to_filename()