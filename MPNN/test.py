import pytest
import deepchem as dc
import tempfile
import pandas as pd
import numpy as np
import os
print(os.getcwd())
from dataset_mpnn import get_dataset, get_class_imbalance_ratio

try:
    import torch
    has_torch = True
except:
    has_torch = False
    

def test_custom_mpnn_model_classification(nb_epoch):
    """
    """
    torch.manual_seed(0)

    # load sample dataset
    # dataset, _ = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    dataset, _ = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')

    randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
    train_dataset, test_dataset = randomstratifiedsplitter.train_test_split(dataset, frac_train = 0.8, seed = 0)
    train_ratios = get_class_imbalance_ratio(pd.DataFrame(train_dataset.y))

    # initialize the model
    from custom_mpnn import CustomMPNNModel
    from featurizer import GraphConvConstants

    model = CustomMPNNModel(n_tasks = 138,
                            batch_size=32,
                            learning_rate=0.001,
                            class_imbalance_ratio = train_ratios,
                            node_out_feats = 50,
                            edge_hidden_feats = 120,
                            edge_out_feats = 50,
                            num_step_message_passing = 2,
                            mode = 'classification',
                            number_atom_features = GraphConvConstants.ATOM_FDIM,
                            number_bond_features = GraphConvConstants.BOND_FDIM,
                            n_classes = 1,
                            ffn_hidden_list= [64, 64],
                            ffn_embeddings = 256,
                            ffn_activation = 'leakyrelu',
                            ffn_dropout_p = 0.1,
                            ffn_dropout_at_input_no_act = True,
                            weight_decay = 1e-5,
                            self_loop = False,
                            log_frequency = 10)

    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')
    # test
    for epoch in range(1, nb_epoch+1):
        loss = model.fit(
          train_dataset,
          nb_epoch=1,
          max_checkpoints_to_keep=1,
          deterministic=False,
          restore=epoch > 1)
        train_scores = model.evaluate(train_dataset, [metric], n_classes=2)
        test_scores = model.evaluate(test_dataset, [metric], n_classes=2)
        print(f"epoch {epoch}/{nb_epoch} ; loss = {loss}; train_scores = {train_scores}; test_scores = {test_scores}")


if __name__ == "__main__":
    test_custom_mpnn_model_classification(10)