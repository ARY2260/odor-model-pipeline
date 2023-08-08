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
from utils.metric_func import macro_averaged_auc_roc_eval
import torch
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def save_checkpoint(self,
                        max_checkpoints_to_keep: int = 5,
                        model_dir = None, ckpt_name='best_checkpoint') -> None:
        """Save a checkpoint to disk.

        Usually you do not need to call this method, since fit() saves checkpoints
        automatically.  If you have disabled automatic checkpointing during fitting,
        this can be called to manually write checkpoints.

        Parameters
        ----------
        max_checkpoints_to_keep: int
            the maximum number of checkpoints to keep.  Older checkpoints are discarded.
        model_dir: str, default None
            Model directory to save checkpoint to. If None, revert to self.model_dir
        """
        self._ensure_built()
        if model_dir is None:
            model_dir = self.model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the checkpoint to a file.

        data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self._pytorch_optimizer.state_dict(),
            'global_step': self._global_step
        }
        temp_file = os.path.join(model_dir, f'{ckpt_name}.pt')
        torch.save(data, temp_file)

        # Rename and delete older files.

        paths = [
            os.path.join(model_dir, f'{ckpt_name}%d.pt' % (i + 1))
            for i in range(max_checkpoints_to_keep)
        ]
        if os.path.exists(paths[-1]):
            os.remove(paths[-1])
        for i in reversed(range(max_checkpoints_to_keep - 1)):
            if os.path.exists(paths[i]):
                os.rename(paths[i], paths[i + 1])
        os.rename(temp_file, paths[0])

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
        # if metric is None:
            # metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.51, classification_handling_mode='threshold')
        for fold_num, (train_dataset, valid_dataset) in enumerate(self.folds_list):
            logger.info("Fitting model %d/%d folds" %
                        (fold_num + 1, self.n_folds))
            hp_str = f"fold_{fold_num + 1}_trial_count_{model_params['trial_count']}_{str(datetime.now())}"

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
            error = ""
            try:
                for epoch in tqdm(range(1, max_epoch+1)):
                    loss = model.fit(
                            train_dataset,
                            nb_epoch=1,
                            max_checkpoints_to_keep=1,
                            deterministic=False,
                            restore= epoch>1)

                    train_scores = macro_averaged_auc_roc_eval(dataset=train_dataset, model=model)
                    valid_scores = macro_averaged_auc_roc_eval(dataset=valid_dataset, model=model)
                    if valid_scores > best_val_score:
                        best_val_score = valid_scores
                        best_train_score = train_scores
                        # save_checkpoint(model, 1, None, f'best_chkpt_{fold_num}_')
                    logger.info(
                        f"epoch {epoch}/{max_epoch} ; loss = {loss}; train_scores = {train_scores}; test_scores = {valid_scores}")

                # best_train_score = model.evaluate(
                #         train_dataset, [metric], n_classes=2)['roc_auc_score']
                # best_val_score = model.evaluate(
                #         valid_dataset, [metric], n_classes=2)['roc_auc_score']
            # Not all models have nb_epoch
            except Exception as e:
                error = f"Training error: {e}"

            all_folds_train_scores.append(best_train_score)
            all_folds_val_scores.append(best_val_score)
            del model
            torch.cuda.empty_cache()
        mean_train_score = np.asarray(all_folds_train_scores).mean()
        mean_val_score = np.asarray(all_folds_val_scores).mean()
        logger.info("Results:")
        logger.info(f"hyperparameters: {str(model_params)}")
        logger.info(f"fold train scores: {all_folds_train_scores}")
        logger.info(f"fold validation scores: {all_folds_val_scores}")
        logger.info(f"mean train score: {mean_train_score}")
        logger.info(f"mean validation score: {mean_val_score}")
        return mean_train_score, mean_val_score, error


#%%
# import torch
# torch.cuda.get_device_name()
# device_list = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
#%%

def random_search_cv(n_folds=2, n_trials=1, logdir='./models', max_epoch=10):
    # Dataset
    # dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')
    dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4983.csv')
    n_tasks = len(dataset.tasks)
    n_folds = n_folds
    n_trials = n_trials
    logdir=logdir
    max_epoch = max_epoch

    # Metric
    # metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')

    model_builder = lambda **params: CustomMPNNModel(n_tasks=n_tasks,
                                         mode='classification',
                                         number_atom_features=GraphConvConstants.ATOM_FDIM,
                                         number_bond_features=GraphConvConstants.BOND_FDIM,
                                         n_classes=1,
                                         ffn_embeddings=256,
                                         ffn_dropout_at_input_no_act=False,
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

#     trials_dict = {
#     "trial_1": {
#         "batch_size": 128,
#         "node_out_feats": 100,
#         "edge_hidden_feats": 75,
#         "edge_out_feats": 100,
#         "num_step_message_passing": 5,
#         "message_aggregator_type" : "sum",
#         "readout_type" : "set2set",
#         "num_step_set2set" : 3,
#         "num_layer_set2set" : 2,
#         "ffn_hidden_list": [
#             392,
#             392
#         ],
#         "ffn_activation": "relu",
#         "ffn_dropout_p": 0.12,
#         "weight_decay": 1e-5,
#         "learning_rate": 0.001,
#         "optimizer_name": "adam"
#     }
# }

    logger.info("Starting random search crosss validation:")
    best_train_score = 0
    best_validation_score = 0
    best_hyperparams = {}
    all_scores = {}
    for trial_count, model_params in tqdm(trials_dict.items()):
        logger.info(f"{trial_count} starting:")
        model_params['trial_count'] = trial_count
        trial_start_time = datetime.now()
        mean_train_score, mean_val_score, error = cv.cross_validation(model_params=model_params, logdir=logdir, max_epoch=max_epoch)
        trial_end_time = datetime.now()
        
        all_scores[trial_count] = {'mean_train_score':mean_train_score, 'mean_val_score':mean_val_score}
        
        current_date_time = str(datetime.now())
        log_file = os.path.join(logdir, f'results_{trial_count}_trial_{current_date_time}.txt')
        with open(log_file, 'w+') as f:
            f.write("Hyperparameters dictionary %s\n" % str(model_params))
            f.write("validation score %f\n" % mean_val_score)
            f.write("train_score: %f\n" % mean_train_score)
            f.write("trial_time: %s\n" % str(trial_end_time-trial_start_time))
            f.write("error: %s\n" % error)

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
#%%
# if __name__ == "__main__":
random_search_cv(max_epoch=2, n_trials=2)

# %%
