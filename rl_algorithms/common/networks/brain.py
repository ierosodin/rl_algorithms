# -*- coding: utf-8 -*-
"""Brain module for backbone & head holder.

- Authors: Euijin Jeong & Kyunghwan Kim
- Contacts: euijin.jeong@medipixel.io
            kh.kim@medipixel.io
"""

import torch
import torch.nn as nn

from rl_algorithms.common.helper_functions import identity
from rl_algorithms.dqn.networks import IQNMLP
from rl_algorithms.registry import build_backbone, build_head
from rl_algorithms.utils.config import ConfigDict


class Brain(nn.Module):
    """Class for holding backbone and head networks."""

    def __init__(
        self, backbone_cfg: ConfigDict, head_cfg: ConfigDict,
    ):
        """Initialize."""
        super(Brain, self).__init__()
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
        self.head = build_head(head_cfg)

    def forward(self, x: torch.Tensor):
        """Forward method implementation. Use in get_action method in agent."""
        x = self.backbone(x)
        x = self.head(x)

        return x

    def forward_(self, x: torch.Tensor, n_tau_samples: int = None):
        """Get output value for calculating loss."""
        x = self.backbone(x)
        if isinstance(self.head, IQNMLP):
            x = self.head.forward_(x, n_tau_samples)
        else:
            x = self.head.forward_(x)
        return x

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]


class GRUBrain(Brain):
    """Class for holding backbone and head networks."""

    def __init__(
        self, backbone_cfg: ConfigDict, head_cfg: ConfigDict,
    ):
        """Initialize."""
        super(GRUBrain, self).__init__(backbone_cfg, head_cfg)
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
        self.gru = nn.GRU(
            head_cfg.configs.input_size,
            head_cfg.configs.rnn_hidden_size,
            batch_first=True,
        )
        head_cfg.configs.input_size = head_cfg.configs.rnn_hidden_size
        self.head = build_head(head_cfg)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor):
        """Forward method implementation. Use in get_action method in agent."""
        x = self.backbone(x)
        if len(x.shape) == 1:
            x = x.reshape(1, 1, -1)
        hidden = torch.transpose(hidden, 0, 1)
        x, hidden = self.gru(x, hidden)
        x = self.head(x)

        return x, hidden

    def forward_(
        self, x: torch.Tensor, hidden: torch.Tensor, n_tau_samples: int = None
    ):
        """Get output value for calculating loss."""
        x = self.backbone(x)
        if len(x.shape) == 1:
            x = x.reshape(1, 1, -1)
        hidden = torch.transpose(hidden, 0, 1)
        x, hidden = self.gru(x, hidden)
        if isinstance(self.head, IQNMLP):
            x = self.head.forward_(x, n_tau_samples)
        else:
            x = self.head.forward_(x)
        return x, hidden

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
