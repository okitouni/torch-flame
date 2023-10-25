import torch
import mup
import torch.nn.functional as F
from typing import Optional, Union
from torch import nn
from typing import Iterable


class MLPEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, hidden_dims, output_dim):
        super().__init__()
        hidden_dims = (
            [hidden_dims]
            if not isinstance(hidden_dims, Iterable)
            else list(hidden_dims)
        )
        self.embedding = torch.nn.Embedding(vocab_size, hidden_dims[0])
        hidden_dims[0] = hidden_dims[0] * 2
        self.mlp = MLP(hidden_dims[0], hidden_dims[1:], output_dim)

    def forward(self, x):
        x = self.embedding(x).flatten(1)
        x = self.mlp(x)
        return x

    def update_readout(self):
        readout_shape = self.mlp.readout.weight.shape
        self.mlp.readout = mup.MuReadout(
            readout_shape[1], readout_shape[0], readout_zero_init=True
        )
        return (self.mlp.readout,)


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim) -> None:
        super().__init__()
        hidden_dims = (
            [hidden_dims]
            if not isinstance(hidden_dims, Iterable)
            else list(hidden_dims)
        )
        widths = [input_dim] + hidden_dims + [output_dim]
        self.act = F.leaky_relu
        self.layers = torch.nn.ModuleList()
        for i in range(len(widths) - 2):
            self.layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
        self.readout = torch.nn.Linear(widths[-2], widths[-1])

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        x = self.readout(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        d_model,
        hidden_dim=None,
        activation: Optional[Union[callable, str]] = "leaky_relu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model
        self.fc1 = torch.nn.Linear(d_model, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, d_model)
        self.act = (
            activation
            if callable(activation)
            else getattr(torch.nn.functional, activation)
        )

    def forward(self, x):
        x = self.fc2(self.act(self.fc1(x))) + x
        return x


class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim) -> None:
        super().__init__()
        hidden_dims = (
            [hidden_dims]
            if not isinstance(hidden_dims, Iterable)
            else list(hidden_dims)
        )
        widths = [input_dim] + hidden_dims + [output_dim]
        self.layers = torch.nn.ModuleList([nn.Linear(widths[0], widths[1])])
        for i in range(1, len(widths) - 1):
            assert (
                widths[i] == widths[1]
            ), "ResMLP requires all hidden layers to have the same width"
            self.layers.append(ResBlock(widths[i]))
        self.readout = torch.nn.Linear(widths[-2], widths[-1])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.readout(x)
        return x


def get_optimizer(model, lr, use_mup=False,**kwargs):
    if use_mup:
        return mup.MuAdam(model.parameters(), lr=lr, **kwargs)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
