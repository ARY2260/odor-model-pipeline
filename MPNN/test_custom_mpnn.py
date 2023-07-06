#%%
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
#%%
def test_custom_mpnn_model_classification(nb_epoch):
    """
    """
    torch.manual_seed(0)

    # load sample dataset
    dataset, class_imbalance_ratio = get_dataset(csv_path='assets/GS_LF_sample100.csv')
    # dataset, class_imbalance_ratio = get_dataset(csv_path='./../curated_GS_LF_merged_4984.csv')

    # initialize the model
    from custom_mpnn import CustomMPNNModel
    from featurizer import GraphConvConstants

    model = CustomMPNNModel(n_tasks = 6,
                        batch_size=10,
                        learning_rate=0.001,
                        class_imbalance_ratio = class_imbalance_ratio,
                        node_out_feats = 10,
                        edge_hidden_feats = 12,
                        edge_out_feats = 10,
                        num_step_message_passing = 2,
                        mode = 'classification',
                        number_atom_features = GraphConvConstants.ATOM_FDIM,
                        number_bond_features = GraphConvConstants.BOND_FDIM,
                        n_classes = 1,
                        ffn_hidden_list= [6, 6],
                        ffn_embeddings = 25,
                        ffn_activation = 'relu',
                        ffn_dropout_p = 0.2,
                        ffn_dropout_at_input_no_act = True,
                        weight_decay = 1e-5,
                        self_loop = False)

    # overfit test
    model.fit(dataset, nb_epoch=nb_epoch)
    metric = dc.metrics.Metric(dc.metrics.roc_auc_score, threshold_value=0.5, classification_handling_mode='threshold')
    scores = model.evaluate(dataset, [metric], n_classes=2)
    print(scores['roc_auc_score'])
#%%
test_custom_mpnn_model_classification(1)
#%%
for count, i in enumerate(range(10, 200, 20)):
    print(f"Iteration : {count+1}, epochs: {i}")
    try:
        test_custom_mpnn_model_classification(i)
    except Exception as e:
        print(e)
    print()
#%%

# def test_dmpnn_model_reload():
#     """
#   Test DMPNNModel class for reloading the model
#   """
#     torch.manual_seed(0)

#     # load sample dataset
#     dir = os.path.dirname(os.path.abspath(__file__))
#     input_file = os.path.join(dir, 'assets/freesolv_sample_5.csv')
#     loader = dc.data.CSVLoader(tasks=['y'],
#                                feature_field='smiles',
#                                featurizer=dc.feat.DMPNNFeaturizer())
#     dataset = loader.create_dataset(input_file)

#     # initialize the model
#     from deepchem.models.torch_models.dmpnn import DMPNNModel
#     model_dir = tempfile.mkdtemp()
#     model = DMPNNModel(model_dir=model_dir, batch_size=2)

#     # fit the model
#     model.fit(dataset, nb_epoch=10)

#     # reload the model
#     reloaded_model = DMPNNModel(model_dir=model_dir, batch_size=2)
#     reloaded_model.restore()

#     orig_predict = model.predict(dataset)
#     reloaded_predict = reloaded_model.predict(dataset)
#     assert np.all(orig_predict == reloaded_predict)


# %%
