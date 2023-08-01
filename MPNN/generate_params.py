#%%
import numpy as np
import json
import os
import itertools
from typing import Dict, List, Any


N_TRIALS = 30

PARAMS_DICT = {'batch_size': [128, 256],
               'node_out_feats': list(range(1, 500, 50)),
               'edge_hidden_feats': list(range(1, 500, 50)),
               'edge_out_feats': list(range(1, 500, 50)),
               'num_step_message_passing': list(range(1, 7, 1)),
               'ffn_hidden_list': [[64],
                                   [256],
                                   [392],
                                   [512],
                                   [64, 64],
                                   [64, 256],],
               'ffn_activation': ['relu', 'leakyrelu'],
               'ffn_dropout_p': np.linspace(0.05, 0.5, num=5).tolist(),
               'weight_decay': [0.001, 0.0001, 1e-05, 1e-06],
               'learning_rate': np.linspace(0.0005, 0.01, num=10).tolist(),
               'optimizer_name': ['adam']}


def generate_random_hyperparam_values(params_dict: Dict,
                                          n: int) -> List[Dict[str, Any]]:
    """Generates `n` random hyperparameter combinations of hyperparameter values
    Parameters
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


def generate_hyperparams(params_dict=PARAMS_DICT, n_trials=N_TRIALS, dir=None):
    """
    Generate hyperparams for random trials
    """
    hyperparameter_combs = generate_random_hyperparam_values(params_dict=params_dict, n=n_trials)

    trials_dict = {}
    for count, params in enumerate(hyperparameter_combs):
        trials_dict[f'trial_{count+1}'] = params

    file_name = f"{n_trials}_trials_params.json"
    if dir is None:
        cwd = os.getcwd()
        file_path = os.path.join(cwd, file_name)
    else:
        file_path = os.path.join(dir, file_name)

    with open(file_path, "w") as json_file:
        json.dump(trials_dict, json_file, indent=4)
    
    return trials_dict, file_path
#%%
# generate_hyperparams()
# %%
