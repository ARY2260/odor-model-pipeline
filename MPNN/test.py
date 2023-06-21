import pytest
import deepchem as dc
import tempfile
import numpy as np
import os
print(os.getcwd())
from dataset_mpnn import get_dataset

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
    # dataset, class_imbalance_ratio = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    dataset, class_imbalance_ratio = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')

    # initialize the model
    from custom_mpnn import CustomMPNNModel
    from featurizer import GraphConvConstants

    model = CustomMPNNModel(n_tasks = 138,
                            batch_size=32,
                            learning_rate=0.001,
                            class_imbalance_ratio = class_imbalance_ratio,
                            node_out_feats = 100,
                            edge_hidden_feats = 120,
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
          dataset,
          nb_epoch=1,
          max_checkpoints_to_keep=1,
          deterministic=False,
          restore=epoch > 1)
        scores = model.evaluate(dataset, [metric], n_classes=2)
        print(f"epoch {epoch}/{nb_epoch} ; loss = {loss}; auc_roc = {scores}")


if __name__ == "__main__":
    test_custom_mpnn_model_classification(10)