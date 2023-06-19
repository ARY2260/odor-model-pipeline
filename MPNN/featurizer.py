from rdkit import Chem
import numpy as np
import logging
from typing import List, Tuple, Union, Dict, Set, Sequence, Optional
from deepchem.utils.typing import RDKitAtom, RDKitMol, RDKitBond

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData

from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot

logger = logging.getLogger(__name__)


class GraphConvConstants(object):
    """
    A class for holding featurization parameters.
    """

    MAX_ATOMIC_NUM = 100
    ATOM_FEATURES: Dict[str, List[int]] = {
        'valence': [0, 1, 2, 3, 4, 5, 6],
        'degree': [0, 1, 2, 3, 4, 5],
        'num_Hs': [0, 1, 2, 3, 4],
        'formal_charge': [-1, -2, 1, 2, 0],
        'atomic_num': list(range(MAX_ATOMIC_NUM)),
    }
    ATOM_FEATURES_HYBRIDIZATION: List[str] = [
        "SP", "SP2", "SP3", "SP3D", "SP3D2"
    ]
    # Dimension of atom feature vector
    ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()
                   ) + len(ATOM_FEATURES_HYBRIDIZATION) + 1
    # len(choices) +1 and len(ATOM_FEATURES_HYBRIDIZATION) +1 to include room for unknown set
    # + 2 at end for is_in_aromatic and mass
    BOND_FDIM = 6


def get_atomic_num_one_hot(atom: RDKitAtom,
                           allowable_set: List[int],
                           include_unknown_set: bool = True) -> List[float]:
    """Get a one-hot feature about atomic number of the given atom.

    Parameters
    ---------
    atom: RDKitAtom
        RDKit atom object
    allowable_set: List[int]
        The range of atomic numbers to consider.
    include_unknown_set: bool, default False
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of atomic number of the given atom.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    """
    return one_hot_encode(atom.GetAtomicNum() - 1, allowable_set,
                          include_unknown_set)


def get_atom_total_valence_one_hot(
        atom: RDKitAtom,
        allowable_set: List[int],
        include_unknown_set: bool = True) -> List[float]:
    """Get an one-hot feature of valence of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
        RDKit atom object
    allowable_set: List[int]
        Atom total valence to consider.
    include_unknown_set: bool, default True
        If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
        A one-hot vector of valence an atom has.
        If `include_unknown_set` is False, the length is `len(allowable_set)`.
        If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    """
    return one_hot_encode(atom.GetTotalValence(), allowable_set,
                          include_unknown_set)


def atom_features(atom: RDKitAtom) -> Sequence[Union[bool, int, float]]:

    if atom is None:
        features: Sequence[Union[bool, int,
                                 float]] = [0] * GraphConvConstants.ATOM_FDIM

    else:
        features = []
        features += get_atom_total_valence_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['valence'])
        features += get_atom_total_degree_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['degree'])
        features += get_atom_total_num_Hs_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['num_Hs'])
        features += get_atom_formal_charge_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['formal_charge'])
        features += get_atomic_num_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES['atomic_num'])
        features += get_atom_hybridization_one_hot(
            atom, GraphConvConstants.ATOM_FEATURES_HYBRIDIZATION, True)
        features = [int(feature) for feature in features]
    return features


def bond_features(bond: RDKitBond) -> Sequence[Union[bool, int, float]]:

    if bond is None:
        b_features: Sequence[Union[
            bool, int, float]] = [1] + [0] * (GraphConvConstants.BOND_FDIM - 1)

    else:
        bt = bond.GetBondType()
        b_features = [
            0,
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing()
        ]

    return b_features


class CustomFeaturizer(MolecularFeaturizer):

    def __init__(self, is_adding_hs = False):
        self.is_adding_hs = is_adding_hs
        super().__init__()

    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Construct edge (bond) index

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        edge_index: np.ndarray
            Edge (Bond) index

        """
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> GraphData:
        """Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: RDKitMol
            RDKit mol object.

        Returns
        -------
        graph: GraphData
            A molecule graph object with features:
            - node_features: Node feature matrix with shape [num_nodes, num_node_features]
            - edge_index: Graph connectivity in COO format with shape [2, num_edges]
            - edge_features: Edge feature matrix with shape [num_edges, num_edge_features]
        """
        if isinstance(datapoint, Chem.rdchem.Mol):
            if self.is_adding_hs:
                datapoint = Chem.AddHs(datapoint)
        else:
            raise ValueError(
                "Feature field should contain smiles for featurizer!")

        # get atom features
        f_atoms: np.ndarray = np.asarray(
            [atom_features(atom) for atom in datapoint.GetAtoms()], dtype=float)

        # get edge(bond) features
        f_bonds_list = []
        for bond in datapoint.GetBonds():
            b_feat = 2 * [bond_features(bond)]
            f_bonds_list.extend(b_feat)
        f_bonds: np.ndarray = np.asarray(f_bonds_list, dtype=float)

        # get edge index
        edge_index: np.ndarray = self._construct_bond_index(datapoint)

        return GraphData(node_features=f_atoms,
                         edge_index=edge_index,
                         edge_features=f_bonds)

# # %%
# import torch
# featurizer = CustomFeaturizer()
# graph = featurizer.featurize('O=C=O')[0]
# g = graph.to_dgl_graph(self_loop = False)

# bond_ft = torch.Tensor([[10.],[20.],[30.],[40.]])
# g.edata['edge_attr'] = bond_ft
# embeddings = torch.Tensor([[1.,2.], [1., 2.], [9., 11.]])
# g.ndata['emb'] = embeddings
# def message_func(edges):
#     src_msg = torch.cat((edges.src['emb'], edges.data['edge_attr']), dim=1)
#     return {'src_msg': src_msg}

# def reduce_func(nodes):
#     src_msg_sum = torch.sum(nodes.mailbox['src_msg'], dim=1)
#     return {'src_msg_sum': src_msg_sum}
# g.send_and_recv(g.edges(), message_func=message_func, reduce_func=reduce_func)
# molecule_hidden_state: torch.Tensor = torch.sum(g.ndata['src_msg_sum'], dim=0)
# # tensor([ 20.,  26., 100.]) required

# # %%
