import joblib
import json
import torch
import torch.nn as nn
import deepchem as dc
import logging
import os
import tempfile
from tqdm import tqdm
from custom_mpnn import CustomMPNNModel
from featurizer import GraphConvConstants
from dataset_mpnn import get_class_imbalance_ratio, get_dataset
# from typing import Dict, List, Optional, Tuple, Any, Callable
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename
import numpy as np
import pandas as pd
from datetime import datetime
from generate_params import generate_hyperparams
from random_search_cv import CV

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def multi_gpu_random_search_cv(dataset=None, n_folds=5, n_trials=30, max_epoch=500, n_jobs=3, logdir='./models'):
    """
    """
    device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

    if dataset is None:
        dataset, _ = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    n_tasks = len(dataset.tasks)
    n_folds = n_folds
    n_trials = n_trials
    max_epoch = max_epoch
    n_jobs = n_jobs
    logdir=logdir

    # Metric
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')

    model_builder = lambda **params: CustomMPNNModel(n_tasks=n_tasks,
                                         mode='classification',
                                         number_atom_features=GraphConvConstants.ATOM_FDIM,
                                         number_bond_features=GraphConvConstants.BOND_FDIM,
                                         n_classes=1,
                                         ffn_embeddings=256,
                                         ffn_dropout_at_input_no_act=True,
                                         self_loop=False,
                                         **params)

    cv = CV(model_builder=model_builder, n_folds=n_folds)
    cv.generate_folds(dataset=dataset, splitter='deepchem')

    # trials_dict, _ = generate_hyperparams(n_trials=n_trials)

    file_name = f"{n_trials}_trials_params.json"
    cwd = os.getcwd()
    file_path = os.path.join(cwd, file_name)
    
    with open(file_path, 'r') as json_file:
        trials_dict = json.load(json_file)

    def wrapper(job_index, model_params):
        device = device_list[job_index % len(device_list)]
        print(f"Job: {job_index+1} on device: {device}")

        model_params['device'] = device
        mean_train_score, mean_val_score = cv.cross_validation(model_params=model_params, logdir=logdir, max_epoch=max_epoch, metric=metric)
        scores = {}
        scores[job_index] = {'params': model_params, 'mean_train_score':mean_train_score, 'mean_val_score':mean_val_score}
        return scores

    all_scores = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(wrapper)(job_index, model_params)
        for job_index, model_params in tqdm(enumerate(trials_dict.values()))
    )

    current_date_time = str(datetime.now())
    all_scores_file = os.path.join(logdir, f'all_scores_{len(trials_dict.items())}_trials_{current_date_time}.json')

    with open(all_scores_file, "w") as json_file:
        json.dump(all_scores, json_file, indent=4)

    best_train_score = 0
    best_validation_score = 0
    best_hyperparams = {}

    for jobs in all_scores:
        scores = list(jobs.values())[0]
        params = scores['params']
        mean_train_score = scores['mean_train_score']
        mean_val_score = scores['mean_val_score']

        if mean_val_score > best_validation_score:
            best_train_score = mean_train_score
            best_validation_score = mean_val_score
            best_hyperparams = params
    
    logger.info("Best hyperparameters: %s" % str(best_hyperparams))
    logger.info("best train_score: %f" % best_train_score)
    logger.info("best validation_score: %f" % best_validation_score)
    
    current_date_time = str(datetime.now())
    results_file = os.path.join(logdir, f'results_{len(trials_dict.items())}_trials_{current_date_time}.txt')
    
    with open(results_file, 'w+') as f:
        f.write("Best Hyperparameters dictionary %s\n" %
                str(best_hyperparams))
        f.write("Best validation score %f\n" % best_validation_score)
        f.write("Best train_score: %f\n" % best_train_score)

if __name__ == "__main__":
    # TEST
    multi_gpu_random_search_cv(dataset=None, n_folds=2, n_trials=3, max_epoch=1, n_jobs=3, logdir='./models')

    # 30 TRIALS
    # dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')
    # multi_gpu_random_search_cv(dataset=dataset, n_folds=5, n_trials=30, max_epoch=500, n_jobs=3, logdir='./models')
    print('multi_gpu_random_search_cv completed successfully!')