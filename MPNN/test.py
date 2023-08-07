import pytest
import deepchem as dc
import tempfile
import pandas as pd
import numpy as np
import os
print(os.getcwd())
from dataset_mpnn import get_dataset, get_class_imbalance_ratio
from deepchem.models.optimizers import ExponentialDecay
from datetime import datetime
from utils.splitter import iterative_train_test_split
from utils.pom_frame import pom_frame
from utils.metric_func import macro_averaged_auc_roc_eval
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    import torch
    has_torch = True
except:
    has_torch = False


def test_custom_mpnn_model_classification(nb_epoch):
    """
    """
    start_time = datetime.now()
    torch.manual_seed(1)

    # load sample dataset
    # dataset, _ = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    # dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')
    dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4983.csv')

    n_tasks = len(dataset.tasks)
    # randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
    # train_dataset, test_dataset, valid_dataset = randomstratifiedsplitter.train_valid_test_split(dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1, seed = 1)
    train_dataset, other_dataset = iterative_train_test_split(dataset, test_size= 0.2, random_state=None, train_dir=None, test_dir=None)
    valid_dataset, test_dataset = iterative_train_test_split(other_dataset, test_size= 0.5, random_state=None, train_dir=None, test_dir=None)
    train_ratios = get_class_imbalance_ratio(pd.DataFrame(train_dataset.y))

    print("train_dataset: ", len(train_dataset))
    print("valid_dataset: ", len(valid_dataset))
    print("test_dataset: ", len(test_dataset))

    # learning_rate = ExponentialDecay(initial_rate=0.001, decay_rate=0.5, decay_steps=32*15, staircase=True)

    learning_rate = 0.001

    # initialize the model
    from custom_mpnn import CustomMPNNModel
    from featurizer import GraphConvConstants

    model = CustomMPNNModel(n_tasks = n_tasks,
                            batch_size=128,
                            learning_rate=learning_rate,
                            class_imbalance_ratio = train_ratios,
                            node_out_feats = 100,
                            edge_hidden_feats = 75,
                            edge_out_feats = 100,
                            num_step_message_passing = 5,
                            mpnn_residual = True,
                            message_aggregator_type = 'sum',
                            mode = 'classification',
                            number_atom_features = GraphConvConstants.ATOM_FDIM,
                            number_bond_features = GraphConvConstants.BOND_FDIM,
                            n_classes = 1,
                            readout_type = 'set2set',
                            num_step_set2set = 3,
                            num_layer_set2set = 2,
                            ffn_hidden_list= [392, 392],
                            ffn_embeddings = 256,
                            ffn_activation = 'relu',
                            ffn_dropout_p = 0.12,
                            ffn_dropout_at_input_no_act = False,
                            weight_decay = 1e-5,
                            self_loop = False,
                            optimizer_name = 'adam',
                            log_frequency = 32,
                            model_dir = './models')

    # metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5000001, classification_handling_mode='threshold')
    # test
    # pom_frame(model, dataset, 0, './models/frames')
    print("lr: ", model.optimizer.learning_rate)
    for epoch in range(1, nb_epoch+1):
        loss = model.fit(
              train_dataset,
              nb_epoch=1,
              max_checkpoints_to_keep=1,
              deterministic=False,
              restore=epoch>1)
        train_scores = macro_averaged_auc_roc_eval(dataset=train_dataset, model=model)
        valid_scores = macro_averaged_auc_roc_eval(dataset=valid_dataset, model=model)
        print(f"epoch {epoch}/{nb_epoch} ; loss = {loss}; train_scores = {train_scores}; test_scores = {valid_scores}")
        model.save_checkpoint()
        # pom_frame(model, dataset, epoch, './models/frames')
    # print(f"epoch {epoch}/{nb_epoch} ; loss = {loss}")

    # train_scores = model.evaluate(train_dataset, [metric], n_classes=2)
    train_scores = macro_averaged_auc_roc_eval(dataset=train_dataset, model=model)
    # test_scores = model.evaluate(test_dataset, [metric], n_classes=2)
    valid_scores = macro_averaged_auc_roc_eval(dataset=valid_dataset, model=model)
    
    print(f"loss = {loss}; train_scores = {train_scores}; valid_scores = {valid_scores}")
    
    end_time = datetime.now()
    print("time_taken: ", str(end_time-start_time))
    # model.save()
    test_scores = macro_averaged_auc_roc_eval(dataset=test_dataset, model=model)
    print("test_score: ", test_scores)

if __name__ == "__main__":
    test_custom_mpnn_model_classification(100)
