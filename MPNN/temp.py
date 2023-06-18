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
