#%%
import deepchem as dc
import logging
import os
import tempfile
from tqdm import tqdm
from custom_mpnn import CustomMPNNModel
from featurizer import GraphConvConstants
from dataset_mpnn import get_class_imbalance_ratio, get_dataset
from typing import Dict, List, Optional, Tuple, Any, Callable
from deepchem.hyper.base_classes import _convert_hyperparam_dict_to_filename
import numpy as np
import pandas as pd
from datetime import datetime
from generate_params import generate_hyperparams

logger = logging.getLogger(__name__)


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

    def cross_validation(self, model_params, logdir=None, max_epoch=100, metric=None):
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
#%%
random_search_cv()
#%%


# def hyperparam_search(
#     self,
#     params_dict: Dict,
#     train_dataset: Dataset,
#     valid_dataset: Dataset,
#     metric: Metric,
#     output_transformers: List[Transformer] = [],
#     nb_epoch: int = 10,
#     use_max: bool = True,
#     logfile: str = 'results.txt',
#     logdir: Optional[str] = None,
#     **kwargs,
# ) -> Tuple[Model, Dict[str, Any], Dict[str, Any]]:
#     """Perform random hyperparams search according to `params_dict`.

#     Each key of the `params_dict` is a model_param. The
#     values should either be a list of potential values of that hyperparam
#     or a callable which can generate random samples.

#     Parameters
#     ----------
#     params_dict: Dict
#         Maps each hyperparameter name (string) to either a list of possible
#         parameter values or a callable which can generate random samples.
#     train_dataset: Dataset
#         dataset used for training
#     valid_dataset: Dataset
#         dataset used for validation (optimization on valid scores)
#     metric: Metric
#         metric used for evaluation
#     output_transformers: list[Transformer]
#         Transformers for evaluation. This argument is needed since
#         `train_dataset` and `valid_dataset` may have been transformed
#         for learning and need the transform to be inverted before
#         the metric can be evaluated on a model.
#     nb_epoch: int, (default 10)
#         Specifies the number of training epochs during each iteration of optimization.
#         Not used by all model types.
#     use_max: bool, optional
#         If True, return the model with the highest score. Else return
#         model with the minimum score.
#     logdir: str, optional
#         The directory in which to store created models. If not set, will
#         use a temporary directory.
#     logfile: str, optional (default `results.txt`)
#         Name of logfile to write results to. If specified, this is must
#         be a valid file name. If not specified, results of hyperparameter
#         search will be written to `logdir/results.txt`.

#     Returns
#     -------
#     Tuple[`best_model`, `best_hyperparams`, `all_scores`]
#         `(best_model, best_hyperparams, all_scores)` where `best_model` is
#         an instance of `dc.model.Model`, `best_hyperparams` is a
#         dictionary of parameters, and `all_scores` is a dictionary mapping
#         string representations of hyperparameter sets to validation
#         scores.
#     """

#     # hyperparam_list should either be an Iterable sequence or a random sampler with rvs method
#     for hyperparam in params_dict.values():
#         assert isinstance(hyperparam,
#                           collections.abc.Iterable) or callable(hyperparam)

#     if use_max:
#         best_validation_score = -np.inf
#     else:
#         best_validation_score = np.inf

#     best_model = None
#     all_scores = {}

#     if logdir is not None:
#         if not os.path.exists(logdir):
#             os.makedirs(logdir, exist_ok=True)
#         log_file = os.path.join(logdir, logfile)

#     hyperparameter_combs = RandomHyperparamOpt.generate_random_hyperparam_values(
#         params_dict, self.max_iter)

#     for ind, model_params in enumerate(hyperparameter_combs):
#         logger.info("Fitting model %d/%d" % (ind + 1, self.max_iter))
#         logger.info("hyperparameters: %s" % str(model_params))

#         hp_str = _convert_hyperparam_dict_to_filename(model_params)

#         if logdir is not None:
#             model_dir = os.path.join(logdir, hp_str)
#             logger.info("model_dir is %s" % model_dir)
#             try:
#                 os.makedirs(model_dir)
#             except OSError:
#                 if not os.path.isdir(model_dir):
#                     logger.info(
#                         "Error creating model_dir, using tempfile directory"
#                     )
#                     model_dir = tempfile.mkdtemp()
#         else:
#             model_dir = tempfile.mkdtemp()

#         model_params['model_dir'] = model_dir
#         model = self.model_builder(**model_params)

#         # mypy test throws error, so ignoring it in try
#         try:
#             model.fit(train_dataset, nb_epoch=nb_epoch)  # type: ignore
#         # Not all models have nb_epoch
#         except TypeError:
#             model.fit(train_dataset)
#         try:
#             model.save()
#         # Some models autosave
#         except NotImplementedError:
#             pass

#         multitask_scores = model.evaluate(valid_dataset, [metric],
#                                           output_transformers)
#         valid_score = multitask_scores[metric.name]

#         # Update best validation score so far
#         if (use_max and valid_score >= best_validation_score) or (
#                 not use_max and valid_score <= best_validation_score):
#             best_validation_score = valid_score
#             best_hyperparams = model_params
#             best_model = model
#             all_scores[hp_str] = valid_score

#         # if `hyp_str` not in `all_scores`, store it in `all_scores`
#         if hp_str not in all_scores:
#             all_scores[hp_str] = valid_score

#         logger.info("Model %d/%d, Metric %s, Validation set %s: %f" %
#                     (ind + 1, nb_epoch, metric.name, ind, valid_score))
#         logger.info("\tbest_validation_score so far: %f" %
#                     best_validation_score)

#     if best_model is None:

#         logger.info("No models trained correctly.")

#         # arbitrarily return last model trained
#         if logdir is not None:
#             with open(log_file, 'w+') as f:
#                 f.write(
#                     "No model trained correctly. Arbitary models returned")

#         best_model, best_hyperparams = model, model_params
#         return best_model, best_hyperparams, all_scores

#     multitask_scores = best_model.evaluate(train_dataset, [metric],
#                                            output_transformers)
#     train_score = multitask_scores[metric.name]
#     logger.info("Best hyperparameters: %s" % str(best_hyperparams))
#     logger.info("best train_score: %f" % train_score)
#     logger.info("best validation_score: %f" % best_validation_score)

#     if logdir is not None:
#         with open(log_file, 'w+') as f:
#             f.write("Best Hyperparameters dictionary %s\n" %
#                     str(best_hyperparams))
#             f.write("Best validation score %f\n" % best_validation_score)
#             f.write("Best train_score: %f\n" % train_score)
#     return best_model, best_hyperparams, all_scores
