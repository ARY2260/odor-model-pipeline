import torch
import torch.nn as nn


class CustomPositionwiseFeedForward(nn.Module):
    """
    Customised PositionwiseFeedForward from deepchem for hidden layers of variable sizes

    NOTE: len(d_hidden_list) == n_layers - 2
    """

    def __init__(self,
                 d_input: int = 1024,
                 d_hidden_list = [1024],
                 d_output: int = 1024,
                 activation: str = 'leakyrelu',
                 dropout_p: float = 0.0,
                 dropout_at_input_no_act: bool = False):
        """Initialize a PositionwiseFeedForward layer.

        Parameters
        ----------
        d_input: int
            Size of input layer.
        d_hidden: int (same as d_input if d_output = 0)
            Size of hidden layer.
        d_output: int (same as d_input if d_output = 0)
            Size of output layer.
        activation: str
            Activation function to be used. Can choose between 'relu' for ReLU, 'leakyrelu' for LeakyReLU, 'prelu' for PReLU,
            'tanh' for TanH, 'selu' for SELU, 'elu' for ELU and 'linear' for linear activation.
        dropout_p: float
            Dropout probability.
        dropout_at_input_no_act: bool
            If true, dropout is applied on the input tensor. For single layer, it is not passed to an activation function.
        """
        super(CustomPositionwiseFeedForward, self).__init__()

        self.dropout_at_input_no_act: bool = dropout_at_input_no_act

        if activation == 'relu':
            self.activation = nn.ReLU()

        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.1)

        elif activation == 'prelu':
            self.activation = nn.PReLU()

        elif activation == 'tanh':
            self.activation = nn.Tanh()

        elif activation == 'selu':
            self.activation = nn.SELU()

        elif activation == 'elu':
            self.activation = nn.ELU()

        elif activation == "linear":
            self.activation = lambda x: x

        d_output = d_output if d_output != 0 else d_input
        
        # Set n_layers
        if len(d_hidden_list) == 0:
            self.n_layers: int = 1
        else:
            self.n_layers = len(d_hidden_list) + 1

        # Set linear layers
        if self.n_layers == 1:
            self.linears = [nn.Linear(d_input, d_output)]

        else:
            self.linears = [nn.Linear(d_input, d_hidden_list[0])]
            
            for idx in range(1, len(d_hidden_list)):
                self.linears.append(nn.Linear(d_hidden_list[idx-1], d_hidden_list[idx]))
            
            self.linears.append(nn.Linear(d_hidden_list[-1], d_output))

        self.linears = nn.ModuleList(self.linears)
        dropout_layer = nn.Dropout(dropout_p)
        self.dropout_p = nn.ModuleList([dropout_layer for _ in range(self.n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Output Computation for the PositionwiseFeedForward layer.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor.
        """
        if not self.n_layers:
            return x

        if self.n_layers == 1:
            if self.dropout_at_input_no_act:
                return self.linears[0](self.dropout_p[0](x))
            else:
                return self.dropout_p[0](self.activation(self.linears[0](x)))

        else:
            if self.dropout_at_input_no_act:
                x = self.dropout_p[0](x)
            for i in range(self.n_layers - 2):
                x = self.dropout_p[i](self.activation(self.linears[i](x)))

            embeddings = self.linears[self.n_layers - 2](x)
            x = self.dropout_p[self.n_layers - 2](self.activation(embeddings))
            output = self.linears[-1](x)
            return embeddings, output

# # %%
# ffn = CustomPositionwiseFeedForward(d_input = 20,
#                  d_hidden_list = [16, 16],
#                  d_output = 3,
#                  activation = 'leakyrelu',
#                  dropout_p = 0.1,
#                  dropout_at_input_no_act = True)
# # %%
# ffn.linears
# #%%
# from torchsummary import summary
# summary(ffn.cuda(), (20, 16))
# # %%
# from deepchem.models.torch_models import layers
# ffn_og = layers.PositionwiseFeedForward(
#             d_input=20,
#             d_hidden=16,
#             d_output=3,
#             activation='leakyrelu',
#             n_layers=3,
#             dropout_p=0.1,
#             dropout_at_input_no_act=True)
# # %%
# ffn_og.linears
# # %%
# from torchsummary import summary
# summary(ffn_og.cuda(), (20, 16))
# # %%
