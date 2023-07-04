#%%
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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CV:
    """
    K-FOLD CROSS VALIDATION for deepchem models with custom stratification splitting
    """

    def __init__(self, model_builder, n_folds) -> None:
        self.model_builder = model_builder
        self.n_folds = n_folds

    def _deepchem_splitter(self, dataset):
        randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
        return randomstratifiedsplitter.k_fold_split(dataset=dataset, k=self.n_folds)

    def generate_folds(self, dataset, splitter='deepchem'):
        if splitter == 'deepchem':
            self.folds_list = self._deepchem_splitter(dataset)
        elif splitter == 'skmultilearn':
            raise NotImplementedError
        return self.folds_list

    def cross_validation(self, model_params, logdir=None, max_epoch=100, metric=None, device=None):
        logger.info("hyperparameters: %s" % str(model_params))
        all_folds_train_scores = []
        all_folds_val_scores = []
        if metric is None:
            metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')
        for fold_num, (train_dataset, valid_dataset) in enumerate(self.folds_list):
            logger.info("Fitting model %d/%d folds" %
                        (fold_num + 1, self.n_folds))
            hp_str = f"fold_{fold_num + 1}" + \
                _convert_hyperparam_dict_to_filename(model_params)

            if logdir is not None:
                model_dir = os.path.join(logdir, hp_str)
                logger.info("model_dir is %s" % model_dir)
                try:
                    os.makedirs(model_dir)
                except OSError:
                    if not os.path.isdir(model_dir):
                        logger.info(
                            "Error creating model_dir, using tempfile directory"
                        )
                        model_dir = tempfile.mkdtemp()
            else:
                model_dir = tempfile.mkdtemp()

            model_params['model_dir'] = model_dir
            model_params['class_imbalance_ratio'] = get_class_imbalance_ratio(pd.DataFrame(train_dataset.y))
            model = self.model_builder(**model_params)

            if device is not None:
                model.to(device)

            # mypy test throws error, so ignoring it in try
            # model_ckpt_list = []
            best_train_score = 0 # train score for best validation
            best_val_score = 0
            try:
                for epoch in tqdm(range(max_epoch)):
                    loss = model.fit(
                        train_dataset,
                        nb_epoch=1,
                        max_checkpoints_to_keep=1,
                        deterministic=False,
                        restore=epoch > 0)

                #   if (epoch+1) % 5 == 0:
                    # ckpt_save_path = os.path.join(save_dir, f'{hp_str}_epoch_{epoch}_ckpt.pt')
                    # torch.save(model.model.state_dict(), ckpt_save_path)
                    # model_ckpt_list.append(ckpt_save_path)
                #   metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')
                    train_score = model.evaluate(
                        train_dataset, [metric], n_classes=2)
                    val_score = model.evaluate(
                        valid_dataset, [metric], n_classes=2)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_train_score = train_score
                    logger.info(
                        f"epoch {epoch}/{max_epoch} ; loss = {loss} ; train_score = {train_score} ; val_score = {val_score}")

            # Not all models have nb_epoch
            except Exception as e:
                raise Exception(f"Training error: {e}")

            all_folds_train_scores.append(best_train_score)
            all_folds_val_scores.append(best_val_score)
            del model
        mean_train_score = np.asarray(all_folds_train_scores).mean()
        mean_val_score = np.asarray(all_folds_val_scores).mean()
        logger.info("Results:")
        logger.info(f"hyperparameters: {str(model_params)}")
        logger.info(f"fold train scores: {all_folds_train_scores}")
        logger.info(f"fold validation scores: {all_folds_val_scores}")
        logger.info(f"mean train score: {mean_train_score}")
        logger.info(f"mean validation score: {mean_val_score}")
        return mean_train_score, mean_val_score


#%%
import torch
torch.cuda.get_device_name()
device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
#%%

def random_search_cv():
    # Dataset
    dataset, _ = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    n_tasks = len(dataset.tasks)
    n_folds = 2
    n_trials = 2
    logdir='./models'

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
    
    trials_dict, _ = generate_hyperparams(n_trials=n_trials)

    logger.info("Starting random search crosss validation:")
    best_train_score = 0
    best_validation_score = 0
    best_hyperparams = {}
    all_scores = {}
    for trial_count, model_params in tqdm(trials_dict.items()):
        logger.info(f"{trial_count} starting:")
        mean_train_score, mean_val_score = cv.cross_validation(model_params=model_params, logdir=logdir, max_epoch=2, metric=metric)
        all_scores[_convert_hyperparam_dict_to_filename(model_params)] = {'mean_train_score':mean_train_score, 'mean_val_score':mean_val_score}
        if mean_val_score > best_validation_score:
            best_train_score = mean_train_score
            best_validation_score = mean_val_score
            best_hyperparams = model_params
    
    logger.info("Best hyperparameters: %s" % str(best_hyperparams))
    logger.info("best train_score: %f" % best_train_score)
    logger.info("best validation_score: %f" % best_validation_score)
    
    current_date_time = str(datetime.now())
    log_file = os.path.join(logdir, f'results_{n_trials}_trials_{current_date_time}.txt')
    
    with open(log_file, 'w+') as f:
        f.write("Best Hyperparameters dictionary %s\n" %
                str(best_hyperparams))
        f.write("Best validation score %f\n" % best_validation_score)
        f.write("Best train_score: %f\n" % best_train_score)
