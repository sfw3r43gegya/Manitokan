import argparse
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNActorModel(nn.Module):
    """Because all the agents share the same network,
    input_shape=obs_shape+n_actions+n_agents.

    Args:
        input_dim (int): The input dimension.
        fc_hidden_dim (int): The hidden dimension of the fully connected layer.
        rnn_hidden_dim (int): The hidden dimension of the RNN layer.
        n_actions (int): The number of actions.
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super(RNNActorModel, self).__init__()

        self.rnn_hidden_dim = int(args.rnn_hidden_dim)
        self.fc1 = nn.Linear(args.obs_shape, args.fc_hidden_dim)
        self.rnn = nn.GRUCell(input_size=args.fc_hidden_dim,
                              hidden_size=int(args.rnn_hidden_dim))
        self.fc2 = nn.Linear(int(args.rnn_hidden_dim), args.n_actions)
        self.batch_size = int(args.batch_size_run)
        self.sample_size = int(args.batch_size)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(
        self,
        inputs: torch.Tensor = None,
        hidden_state: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.gelu(self.fc1(inputs))

        if hidden_state is not None:
            h_in = hidden_state.reshape(-1, self.rnn_hidden_dim).squeeze()
        else:
            if len(out.shape) == 1:
                indx = 1
            else:
                indx = self.sample_size*self.batch_size
            h_in = torch.zeros(indx,
                               self.rnn_hidden_dim).to(inputs.device).squeeze()

        hidden_state = self.rnn(out, h_in)
        out = self.fc2(hidden_state)  # (batch_size, n_actions)
        return out, hidden_state

    def update(self, model: nn.Module) -> None:
        self.load_state_dict(model.state_dict())


class MultiLayerRNNActorModel(nn.Module):
    """Because all the agents share the same network,
    input_shape=obs_shape+n_actions+n_agents.

    Args:
        input_dim (int): The input dimension.
        fc_hidden_dim (int): The hidden dimension of the fully connected layer.
        rnn_hidden_dim (int): The hidden dimension of the RNN layer.
        n_actions (int): The number of actions.
    """

    def __init__(
        self,
        input_dim: int = None,
        fc_hidden_dim: int = 64,
        rnn_hidden_dim: int = 64,
        rnn_num_layers: int = 2,
        n_actions: int = None,
        **kwargs,
    ) -> None:
        super(MultiLayerRNNActorModel, self).__init__()

        self.rnn_hidden_dim = int(rnn_hidden_dim)
        self.rnn_num_layers = rnn_num_layers

        self.fc1 = nn.Linear(input_dim, fc_hidden_dim)
        self.rnn = nn.GRU(
            input_size=fc_hidden_dim,
            hidden_size=int(rnn_hidden_dim),
            num_layers=rnn_num_layers,
            batch_first=True,
        )
        self.fc2 = nn.Linear(int(rnn_hidden_dim), n_actions)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        # make hidden states on same device as model
        return self.fc1.weight.new(self.rnn_num_layers, batch_size,
                                   self.rnn_hidden_dim).zero_()