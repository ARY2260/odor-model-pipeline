"""
DGL-based MPNN for graph property prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepchem.models.losses import Loss, L2Loss
from deepchem.models.torch_models.torch_model import TorchModel
from deepchem.models.optimizers import Optimizer, Adam
from custom_ffn import CustomPositionwiseFeedForward

from utils.loss_func import CustomMultiLabelLoss
from utils.train_utils import get_optimizer

try:
    import dgl
except:
    raise ImportError('This class requires dgl.')

try:
    from dgllife.model import MPNNGNN
except:
    raise ImportError('This class requires dgllife.')


class CustomMPNN(nn.Module):
    """Model for Graph Property Prediction.

    This model proceeds as follows:

    * Combine latest node representations and edge features in updating node representations,
        which involves multiple rounds of message passing
    * For each graph, compute its representation by combining the representations
        of all nodes in it, which involves a Set2Set layer.
    * Perform the final prediction using an MLP

    Examples
    --------

    >>> import deepchem as dc
    >>> import dgl
    >>> from deepchem.models.torch_models import MPNN
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    >>> graphs = featurizer.featurize(smiles)
    >>> print(type(graphs[0]))
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> dgl_graphs = [graphs[i].to_dgl_graph(self_loop=True) for i in range(len(graphs))]
    >>> # Batch two graphs into a graph of two connected components
    >>> batch_dgl_graph = dgl.batch(dgl_graphs)
    >>> model = MPNN(n_tasks=1, mode='regression')
    >>> preds = model(batch_dgl_graph)
    >>> print(type(preds))
    <class 'torch.Tensor'>
    >>> preds.shape == (2, 1)
    True

    References
    ----------
    .. [1] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.
        "Neural Message Passing for Quantum Chemistry." ICML 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.
    """

    def __init__(self,
                 n_tasks: int,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 3,
                 mode: str = 'regression',
                 number_atom_features: int = 30,
                 number_bond_features: int = 11,
                 n_classes: int = 2,
                 nfeat_name: str = 'x',
                 efeat_name: str = 'edge_attr',
                 ffn_hidden_list = [300],
                 ffn_embeddings: int = 300,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        node_out_feats: int
            The length of the final node representation vectors. Default to 64.
        edge_hidden_feats: int
            The length of the hidden edge representation vectors. Default to 128.
        num_step_message_passing: int
            The number of rounds of message passing. Default to 3.
        mode: str
            The model type, 'classification' or 'regression'. Default to 'regression'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 30.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 11.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 2.
        nfeat_name: str
            For an input graph ``g``, the model assumes that it stores node features in
            ``g.ndata[nfeat_name]`` and will retrieve input node features from that.
            Default to 'x'.
        efeat_name: str
            For an input graph ``g``, the model assumes that it stores edge features in
            ``g.edata[efeat_name]`` and will retrieve input edge features from that.
            Default to 'edge_attr'.
        ffn_hidden_list: list
            list of Sizes of hidden layer in the feed-forward network layer.
        ffn_activation: str
            Activation function to be used in feed-forward network layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
        ffn_dropout_p: float
            Dropout probability for the feed-forward network layer.
        ffn_dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        if mode not in ['classification', 'regression']:
            raise ValueError(
                "mode must be either 'classification' or 'regression'")

        super(CustomMPNN, self).__init__()

        self.n_tasks = n_tasks
        self.mode = mode
        self.n_classes = n_classes
        self.nfeat_name = nfeat_name
        self.efeat_name = efeat_name
        self.ffn_embeddings = ffn_embeddings
        self.ffn_activation = ffn_activation
        self.ffn_dropout_p = ffn_dropout_p
        if mode == 'classification':
            self.ffn_output = n_tasks * n_classes
        else:
            self.ffn_output = n_tasks

        self.mpnn = MPNNGNN(node_in_feats=number_atom_features,
                           node_out_feats=node_out_feats,
                           edge_in_feats=number_bond_features,
                           edge_hidden_feats=edge_hidden_feats,
                           num_step_message_passing=num_step_message_passing)
        
        if ffn_embeddings is not None:
            ffn_hidden_list.append(ffn_embeddings)

        self.ffn: nn.Module = CustomPositionwiseFeedForward(
            d_input=node_out_feats + number_bond_features,
            d_hidden_list=ffn_hidden_list,
            d_output=self.ffn_output,
            activation=ffn_activation,
            dropout_p=ffn_dropout_p,
            dropout_at_input_no_act=ffn_dropout_at_input_no_act)

    def _readout(self, g, atoms_hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Method to execute the readout phase. (compute molecules encodings from atom hidden states)

        Parameters
        ----------
         g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        atoms_hidden_states: torch.Tensor
            Tensor containing atom hidden states.

        Returns
        -------
        molecule_hidden_state: torch.Tensor
            Tensor containing molecule encodings.
        """

        g.ndata['emb'] = atoms_hidden_states
        graphs_list = dgl.unbatch(g=g)
        mol_feat_tensor_list = []
        for graph in graphs_list:
            mol_feat_tensor_list.append(self._readout_per_g(graph))
        
        batch_mol_hidden_states = torch.stack(mol_feat_tensor_list, dim=0)
        return batch_mol_hidden_states  # batch_size x (node_out_feats + bond_dim)
    
    def _readout_per_g(self, g) -> torch.Tensor:
        """
        a reduce-sum across atoms per graph
        
        Parameters
        ----------
         g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        Returns
        -------
        molecule_hidden_state: torch.Tensor
            Tensor containing molecule encodings.
        """
        def message_func(edges):
            src_msg = torch.cat((edges.src['emb'], edges.data['edge_attr']), dim=1)
            return {'src_msg': src_msg}
        
        def reduce_func(nodes):
            src_msg_sum = torch.sum(nodes.mailbox['src_msg'], dim=1)
            return {'src_msg_sum': src_msg_sum}

        g.send_and_recv(g.edges(), message_func=message_func, reduce_func=reduce_func)
        molecule_hidden_state: torch.Tensor = torch.sum(g.ndata['src_msg_sum'], dim=0)
        return molecule_hidden_state  # (node_out_feats + bond_dim)

    def forward(self, g):
        """Predict graph labels

        Parameters
        ----------
        g: DGLGraph
            A DGLGraph for a batch of graphs. It stores the node features in
            ``dgl_graph.ndata[self.nfeat_name]`` and edge features in
            ``dgl_graph.edata[self.efeat_name]``.

        Returns
        -------
        torch.Tensor
            The model output.

        * When self.mode = 'regression',
            its shape will be ``(dgl_graph.batch_size, self.n_tasks)``.
        * When self.mode = 'classification', the output consists of probabilities
            for classes. Its shape will be
            ``(dgl_graph.batch_size, self.n_tasks, self.n_classes)`` if self.n_tasks > 1;
            its shape will be ``(dgl_graph.batch_size, self.n_classes)`` if self.n_tasks is 1.
        torch.Tensor, optional
            This is only returned when self.mode = 'classification', the output consists of the
            logits for classes before softmax.
        """
        node_feats = g.ndata[self.nfeat_name]
        edge_feats = g.edata[self.efeat_name]
        
        node_encodings = self.mpnn(g, node_feats, edge_feats)
        for p in self.mpnn.named_parameters():
            if torch.isnan(p[1]).any():
                print(p[0])
        if torch.isnan(node_encodings).any():
            raise Exception("contains NaN!")
        molecular_encodings = self._readout(g, node_encodings)
        embeddings, out = self.ffn(molecular_encodings)

        if self.mode == 'classification':
            if self.n_tasks == 1:
                logits = out.view(-1, self.n_classes)
            else:
                logits = out.view(-1, self.n_tasks, self.n_classes)
            proba = F.sigmoid(logits) # (batch, n_tasks, classes)
            if self.n_classes == 1:
                proba = proba.squeeze(-1) # (batch, n_tasks)
            return proba, logits, embeddings
        else:
            return out


class CustomMPNNModel(TorchModel):
    """Model for graph property prediction

    This model proceeds as follows:

    * Combine latest node representations and edge features in updating node representations,
        which involves multiple rounds of message passing
    * For each graph, compute its representation by combining the representations
        of all nodes in it, which involves a Set2Set layer.
    * Perform the final prediction using an MLP
    Examples
    --------
    >>> import deepchem as dc
    >>> from deepchem.models.torch_models import MPNNModel
    >>> # preparing dataset
    >>> smiles = ["C1CCC1", "CCC"]
    >>> labels = [0., 1.]
    >>> featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
    >>> X = featurizer.featurize(smiles)
    >>> dataset = dc.data.NumpyDataset(X=X, y=labels)
    >>> # training model
    >>> model = MPNNModel(mode='classification', n_tasks=1,
    ...                  batch_size=16, learning_rate=0.001)
    >>> loss =  model.fit(dataset, nb_epoch=5)

    References
    ----------
    .. [1] Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.
        "Neural Message Passing for Quantum Chemistry." ICML 2017.

    Notes
    -----
    This class requires DGL (https://github.com/dmlc/dgl) and DGL-LifeSci
    (https://github.com/awslabs/dgl-lifesci) to be installed.

    The featurizer used with MPNNModel must produce a GraphData object which should have both 'edge' and 'node' features.
    """

    def __init__(self,
                 n_tasks: int,
                 class_imbalance_ratio = None,
                 node_out_feats: int = 64,
                 edge_hidden_feats: int = 128,
                 num_step_message_passing: int = 3,
                 mode: str = 'regression',
                 number_atom_features: int = 30,
                 number_bond_features: int = 11,
                 n_classes: int = 2,
                 ffn_hidden_list = [300],
                 ffn_embeddings: int = 300,
                 ffn_activation: str = 'relu',
                 ffn_dropout_p: float = 0.0,
                 ffn_dropout_at_input_no_act: bool = True,
                 weight_decay: float = 0.0,
                 self_loop: bool = False,
                 optimizer_name = 'adam',
                 learning_rate: float = 0.001,
                 **kwargs):
        """
        Parameters
        ----------
        n_tasks: int
            Number of tasks.
        node_out_feats: int
            The length of the final node representation vectors. Default to 64.
        edge_hidden_feats: int
            The length of the hidden edge representation vectors. Default to 128.
        num_step_message_passing: int
            The number of rounds of message passing. Default to 3.
        mode: str
            The model type, 'classification' or 'regression'. Default to 'regression'.
        number_atom_features: int
            The length of the initial atom feature vectors. Default to 30.
        number_bond_features: int
            The length of the initial bond feature vectors. Default to 11.
        n_classes: int
            The number of classes to predict per task
            (only used when ``mode`` is 'classification'). Default to 2.
        ffn_hidden_list: list
            list of Sizes of hidden layer in the feed-forward network layer.
        ffn_activation: str
            Activation function to be used in feed-forward network layer.
            Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, and 'elu' for ELU.
        ffn_layers: int
            Number of layers in the feed-forward network layer.
        ffn_dropout_p: float
            Dropout probability for the feed-forward network layer.
        ffn_dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        self_loop: bool
            Whether to add self loops for the nodes, i.e. edges from nodes to themselves.
            Generally, an MPNNModel does not require self loops. Default to False.
        kwargs
            This can include any keyword argument of TorchModel.
        """
        model = CustomMPNN(n_tasks=n_tasks,
                     node_out_feats=node_out_feats,
                     edge_hidden_feats=edge_hidden_feats,
                     num_step_message_passing=num_step_message_passing,
                     mode=mode,
                     number_atom_features=number_atom_features,
                     number_bond_features=number_bond_features,
                     n_classes=n_classes,
                     ffn_hidden_list=ffn_hidden_list,
                     ffn_embeddings=ffn_embeddings,
                     ffn_activation=ffn_activation,
                     ffn_dropout_p=ffn_dropout_p,
                     ffn_dropout_at_input_no_act=ffn_dropout_at_input_no_act)
        if mode == 'regression':
            loss: Loss = L2Loss()
            output_types = ['prediction']
        else:
            class_imbalance_ratio = torch.tensor(class_imbalance_ratio).to('cuda' if torch.cuda.is_available() else 'cpu')
            loss = CustomMultiLabelLoss(class_imbalance_ratio=class_imbalance_ratio)
            output_types = ['prediction', 'loss', 'embedding']

        self.weight_decay = weight_decay

        optimizer = get_optimizer(optimizer_name)
        if isinstance(optimizer, Optimizer):
            optimizer.learning_rate = learning_rate
        else:
            optimizer = None
        
        super(CustomMPNNModel, self).__init__(model,
                                        loss=loss,
                                        output_types=output_types,
                                        optimizer=optimizer,
                                        learning_rate=learning_rate,
                                        **kwargs)

        self._self_loop = self_loop

        self.regularization_loss = self._regularization_loss
    
    def _regularization_loss(self):
        """
        l1 and l2-norm losses for regularization
        """
        l1_regularization = torch.tensor(0., requires_grad=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        l2_regularization = torch.tensor(0., requires_grad=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        for name, param in self.model.named_parameters():
            if 'bias' not in name:
                l1_regularization += torch.norm(param, p=1) #l1
                l2_regularization += torch.norm(param, p=2) #l2
        l1_norm = self.weight_decay * l1_regularization
        l2_norm = self.weight_decay * l2_regularization
        return l1_norm + l2_norm

    def _prepare_batch(self, batch):
        """Create batch data for MPNN.

        Parameters
        ----------
        batch: tuple
            The tuple is ``(inputs, labels, weights)``.

        Returns
        -------
        inputs: DGLGraph
            DGLGraph for a batch of graphs.
        labels: list of torch.Tensor or None
            The graph labels.
        weights: list of torch.Tensor or None
            The weights for each sample or sample/task pair converted to torch.Tensor.
        """

        inputs, labels, weights = batch
        dgl_graphs = [
            graph.to_dgl_graph(self_loop=self._self_loop) for graph in inputs[0]
        ]
        inputs = dgl.batch(dgl_graphs).to(self.device)
        _, labels, weights = super(CustomMPNNModel, self)._prepare_batch(
            ([], labels, weights))
        return inputs, labels, weights
