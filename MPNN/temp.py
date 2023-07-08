#%%
from rdkit import Chem
import pandas as pd

curated_df = pd.read_csv('./../curated_GS_LF_merged_4984.csv')
smiles_list = curated_df['nonStereoSMILES'].to_list()
valence_set = set()
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
      valence_set.update([atom.GetTotalValence()])

# 0, 1, 2, 3, 4, 5, 6
# %%
from rdkit import Chem
import pandas as pd

curated_df = pd.read_csv('./../curated_GS_LF_merged_4984.csv')
smiles_list = curated_df['nonStereoSMILES'].to_list()
atom_num_set = set()
for smiles in smiles_list:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
      atom_num_set.update([atom.GetAtomicNum()])

# max 83
# %%
mol = Chem.MolFromSmiles('CC')
mol2 = Chem.AddHs(mol)

for atom in mol2.GetAtoms():
      print(atom.GetTotalNumHs())
# %%
from custom_mpnn import CustomMPNNModel
from dataset_mpnn import get_dataset
from featurizer import GraphConvConstants

dataset, class_imbalance_ratio = get_dataset()
#%%
model = CustomMPNNModel(n_tasks = 138,
                        batch_size=50,
                        learning_rate=0.001,
                        class_imbalance_ratio = class_imbalance_ratio,
                        node_out_feats = 30,
                        edge_hidden_feats = 12,
                        edge_out_feats = 30,
                        num_step_message_passing = 1,
                        mode = 'classification',
                        number_atom_features = GraphConvConstants.ATOM_FDIM,
                        number_bond_features = GraphConvConstants.BOND_FDIM,
                        n_classes = 1,
                        aggregation = 'sum',
                        aggregation_norm = 0,
                        ffn_hidden_list= [20, 20],
                        ffn_embeddings = 256,
                        ffn_activation = 'relu',
                        ffn_dropout_p = 0.0,
                        ffn_dropout_at_input_no_act = True,
                        weight_decay = 1e-5,
                        self_loop = False)
#%%
loss =  model.fit(dataset, nb_epoch=30)
# %%
from rdkit import Chem
import pandas as pd

curated_df = pd.read_csv('./../curated_GS_LF_merged_4984.csv')
# %%
sample_df = curated_df.sample(100)
sample_df
# %%
encoded = sample_df.drop(columns=['nonStereoSMILES', 'descriptors'])
odors_df = pd.DataFrame(encoded.sum().sort_values(ascending=False), columns=['sum'])
odors_df.query('sum<15')
#%%
cols_to_drop = list(odors_df.query('sum<15')['sum'].index)
#%%
sample_df = sample_df.drop(columns=cols_to_drop)
sample_df = sample_df.reset_index(drop=True)
# %%
sample_df
# %%
sample_df.to_csv('GS_LF_sample100.csv', index=False)
# %%
#%%
from dgllife.model import MPNNGNN
from featurizer import GraphConvConstants

mpnn = MPNNGNN(node_in_feats=GraphConvConstants.ATOM_FDIM,
               node_out_feats=20,
               edge_in_feats=GraphConvConstants.BOND_FDIM,
               edge_hidden_feats=12,
               edge_out_feats = 20,
               num_step_message_passing=1)


from featurizer import CustomFeaturizer

smiles = ["CCC1SC(C)=NC(C)S1", "CCC1=C(O)C(=O)CC1C"]
featurizer = CustomFeaturizer()
feat = featurizer.featurize(smiles)

# graph = feat[0]
# dgl_graph = graph.to_dgl_graph(self_loop=False)

import dgl

dgl_graphs = [
            graph.to_dgl_graph(self_loop=False) for graph in feat
        ]
dgl_batch = dgl.batch(dgl_graphs)

node_feats = dgl_batch.ndata['x']
edge_feats = dgl_batch.edata['edge_attr']

mode_embed = mpnn(dgl_batch, node_feats, edge_feats)
print(mode_embed, mode_embed.shape)
# %%
import torch
ce_loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
# output = torch.Tensor([[[0.6],[0.4],[0.3]], [[0.6],[0.4],[0.3]]])
# labels = torch.Tensor([[[1],[0],[0]], [[0],[1],[1]]])
output = torch.Tensor([[[0.6]]])
labels = torch.Tensor([[[0]]])
def loss(output, labels):
    # Convert (batch_size, tasks, classes) to (batch_size, classes, tasks)
    # CrossEntropyLoss only supports (batch_size, classes, tasks)
    # This is for API consistency
    if len(output.shape) == 3:
        output = output.permute(0, 2, 1)
    if len(labels.shape) == len(output.shape):
        labels = labels.squeeze(-1)
    # handle multilabel
    # output shape => (batch_size, classes=1, tasks)
    # binary_output shape => (batch_size, classes=2, tasks) where now we have (1 - probabilities) for ce loss calculation
    probabilities = output[:, 0, :]
    complement_probabilities = 1 - probabilities
    binary_output1 = torch.stack([probabilities, complement_probabilities], axis=1)
    binary_output2 = torch.stack([complement_probabilities, probabilities], axis=1)
    ce_loss1 = ce_loss_fn(binary_output1, labels.long())
    ce_loss2 = ce_loss_fn(binary_output2, labels.long())

    return ce_loss1, ce_loss2
loss(output, labels)
# %%
import deepchem as dc
from dataset_mpnn import get_dataset, get_class_imbalance_ratio
from featurizer import GraphConvConstants
import numpy as np
#%%
dataset, class_imbalance_ratio = get_dataset()
#%%
randomstratifiedsplitter = dc.splits.RandomStratifiedSplitter()
train_dataset, test_dataset = randomstratifiedsplitter.train_test_split(dataset, frac_train = 0.9, seed = 0)
# %%
folds_list = randomstratifiedsplitter.k_fold_split(dataset=dataset, k=3)
#%%
train_dataset, test_dataset = folds_list[0]
#%%
import pandas as pd

print(train_dataset.y.shape)
train_y_df = pd.DataFrame(train_dataset.y)
test_y_df = pd.DataFrame(test_dataset.y)
train_ratios = get_class_imbalance_ratio(train_y_df)
test_ratios = get_class_imbalance_ratio(test_y_df)
# %%
def get_outliers(df):
    # Calculate outlier boundaries using the interquartile range (IQR)
    Q1 = df.quantile(0.25)[0]
    Q3 = df.quantile(0.75)[0]
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers
    outliers = df[(df < lower_bound) | (df > upper_bound)]
    outlier_count = len(outliers.dropna())

    print(f"Interquartile range = {Q1} % - {Q3} %")
    print(f"Number of outliers:", outlier_count)
    return outliers.dropna().index

def ratio_analysis(new_ratios, original_ratios):
    new_ratios, original_ratios = np.asarray(new_ratios), np.asarray(original_ratios)
    print(f"min ratio in new: {new_ratios.min()} and index: {new_ratios.argmin()}")
    print(f"min ratio in orignal: {original_ratios.min()} and index: {original_ratios.argmin()}")
    print()
    print(f"max ratio in new: {new_ratios.max()} and index: {new_ratios.argmax()}")
    print(f"max ratio in orignal: {original_ratios.max()} and index: {original_ratios.argmax()}")
    diff = new_ratios - original_ratios
    print()
    print(f"deviation range of ratios: ({diff.min()}, {diff.max()})")
    print(f"absolute deviation range of ratios: ({abs(diff).min()}, {abs(diff).max()})")
    percentages = (abs(diff)/class_imbalance_ratio)*100
    print()
    print(f"min absoute percentage change in ratios: {percentages.min()}%")
    print(f"max absoute percentage change in ratios: {percentages.max()}%")
    percentages_df = pd.DataFrame(percentages)
    print()
    print("Box plot analysis of absolute percentage errors:")
    percentages_df.plot.box()
    outlier_indices = get_outliers(percentages_df)
    outliers_names = list(dataset.tasks[list(outlier_indices)])
    outliers_percent_change = list(percentages_df[0][list(outlier_indices)])
    outliers_df = pd.DataFrame([outliers_names,outliers_percent_change])
    print(outliers_df.head())

# %%
ratio_analysis(train_ratios, class_imbalance_ratio)
# %%
ratio_analysis(test_ratios, class_imbalance_ratio)
# %%
ratio_analysis(test_ratios, train_ratios)
# %%
# ['alcoholic', 'coconut', 'creamy', 'lily', 'musk', 'odorless', 'ozone', 'plum', 'radish', 'tea', 'tomato']
# ['alcoholic', 'coconut', 'cortex', 'creamy', 'lily', 'musk', 'ozone', 'plum', 'radish', 'tea', 'tomato']
# ['alcoholic', 'coconut', 'cortex', 'creamy', 'lily', 'musk', 'ozone', 'plum', 'radish', 'tea', 'tomato']
#%%
from skmultilearn.model_selection import IterativeStratification

def iterative_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split

    Parameters
    ----------
    test_size : float, [0,1]
        the proportion of the dataset to include in the test split, the rest will be put in the train set

    random_state : None | int | np.random.RandomState
        the random state seed (optional)

    Returns
    -------
    X_train, y_train, X_test, y_test
        stratified division into train/test split
    """

    stratifier = IterativeStratification(
        n_splits=2,
        order=2,
        sample_distribution_per_fold=[test_size, 1.0 - test_size],
        # shuffle=True,
        random_state=random_state,
    )
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X.iloc[train_indexes, :], y.iloc[train_indexes, :]
    X_test, y_test = X.iloc[test_indexes, :], y.iloc[test_indexes, :]

    return X_train, X_test, y_train, y_test
#%%
X, y = pd.DataFrame(dataset.X), pd.DataFrame(dataset.y)

X_train, X_test, y_train, y_test = iterative_train_test_split(X, y, test_size=0.1, random_state=None)
# %%
train_ratios_is = get_class_imbalance_ratio(y_train)
test_ratios_is = get_class_imbalance_ratio(y_test)
# %%
ratio_analysis(train_ratios_is, class_imbalance_ratio)
# %%
ratio_analysis(test_ratios_is, class_imbalance_ratio)
# %%
ratio_analysis(test_ratios_is, train_ratios_is)
# %%
import logging

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

logger.info("Info")
logger.error("error")
logger.info("info")
print('done')
# %%
import torch
torch.device('cuda:0')
# %%
