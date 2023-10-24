import torch
import mup
import torch.nn.functional as F
from typing import Optional, Union
from torch import nn


class MLP(torch.nn.Module):
    def __init__(self, widths, use_mup=False, savefile=None) -> None:
        super().__init__()
        self.act = F.leaky_relu
        self.layers = torch.nn.ModuleList()
        for i in range(len(widths) - 2):
            self.layers.append(torch.nn.Linear(widths[i], widths[i + 1]))
        readout = mup.MuReadout if use_mup else torch.nn.Linear
        self.readout = readout(widths[-2], widths[-1])
        if use_mup:
            depth = len(widths) - 2
            base_width = 64
            base_widths = [widths[0]] + [base_width] * depth + [widths[-1]]
            delta_widths = [widths[0]] + [base_width + 1] * depth + [widths[-1]]
            base = MLP(base_widths, use_mup=False)
            delta = MLP(delta_widths, use_mup=False)
            mup.set_base_shapes(self, base, delta=delta, savefile=savefile)
            self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        for name, p in self.named_parameters():
            if "bias" in name or "readout" in name:
                mup.init.uniform_(p, 0, 0)
            else:
                # mup.init.kaiming_uniform_(p, a=None)
                mup.init.uniform_(p, -0.1, 0.1)

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


class ResMLP(MLP):
    def __init__(self, widths, use_mup=False, savefile=None) -> None:
        super().__init__(widths, use_mup, savefile)
        self.layers = torch.nn.ModuleList([nn.Linear(widths[0], widths[1])])
        for i in range(1, len(widths) - 1):
            assert (
                widths[i] == widths[1]
            ), "ResMLP requires all hidden layers to have the same width"
            self.layers.append(ResBlock(widths[i]))
        readout = mup.MuReadout if use_mup else torch.nn.Linear
        self.readout = readout(widths[-2], widths[-1])
        if use_mup:
            depth = len(widths) - 2
            base_width = 32
            base_widths = [widths[0]] + [base_width] * depth + [widths[-1]]
            delta_widths = [widths[0]] + [base_width + 1] * depth + [widths[-1]]
            base = ResMLP(base_widths, use_mup=False)
            delta = ResMLP(delta_widths, use_mup=False)
            mup.set_base_shapes(self, base, delta=delta, savefile=savefile)
            self.reset_parameters()


def get_optimizer(model, lr, use_mup=False):
    if use_mup:
        return mup.MuAdam(model.parameters(), lr=lr)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr)
