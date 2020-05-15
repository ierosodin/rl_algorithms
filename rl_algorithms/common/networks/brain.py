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
        self.action_size = head_cfg.configs.output_size
        """Initialize."""
        super(GRUBrain, self).__init__(backbone_cfg, head_cfg)
        if not backbone_cfg:
            self.backbone = identity
            head_cfg.configs.input_size = head_cfg.configs.state_size[0]
            self.fc = nn.Linear(
                head_cfg.configs.input_size, head_cfg.configs.rnn_hidden_size,
            )
            self.gru = nn.GRU(
                head_cfg.configs.rnn_hidden_size
                + self.action_size
                + 1,  # 1 is for prev_reward
                head_cfg.configs.rnn_hidden_size,
                batch_first=True,
            )
        else:
            self.backbone = build_backbone(backbone_cfg)
            head_cfg.configs.input_size = self.calculate_fc_input_size(
                head_cfg.configs.state_size
            )
            self.fc = nn.Linear(
                head_cfg.configs.input_size, head_cfg.configs.rnn_hidden_size,
            )
            self.gru = nn.GRU(
                head_cfg.configs.rnn_hidden_size
                + self.action_size
                + 1,  # 1 is for prev_reward
                head_cfg.configs.rnn_hidden_size,
                batch_first=True,
            )

        head_cfg.configs.input_size = head_cfg.configs.rnn_hidden_size
        self.head = build_head(head_cfg)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, prev_action, prev_reward):
        def infer_leading_dims(tensor, dim):
            """Looks for up to two leading dimensions in ``tensor``, before
            the data dimensions, of which there are assumed to be ``dim`` number.
            For use at beginning of model's ``forward()`` method, which should
            finish with ``restore_leading_dims()`` (see that function for help.)
            Returns:
            lead_dim: int --number of leading dims found.
            T: int --size of first leading dim, if two leading dims, o/w 1.
            B: int --size of first leading dim if one, second leading dim if two, o/w 1.
            shape: tensor shape after leading dims.
            """
            lead_dim = tensor.dim() - dim
            assert lead_dim in (0, 1, 2)
            if lead_dim == 2:
                T, B = tensor.shape[:2]
            else:
                T = 1
                B = 1 if lead_dim == 0 else tensor.shape[0]
            shape = tensor.shape[lead_dim:]
            return lead_dim, T, B, shape

        def restore_leading_dims(tensors, lead_dim, T=1, B=1):
            """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
            leading dimensions, which will become [], [B], or [T,B].  Assumes input
            tensors already have a leading Batch dimension, which might need to be
            removed. (Typically the last layer of model will compute with leading
            batch dimension.)  For use in model ``forward()`` method, so that output
            dimensions match input dimensions, and the same model can be used for any
            such case.  Use with outputs from ``infer_leading_dims()``."""
            is_seq = isinstance(tensors, (tuple, list))
            tensors = tensors if is_seq else (tensors,)
            if lead_dim == 2:  # (Put T dim.)
                tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
            if lead_dim == 0:  # (Remove B=1 dim.)
                assert B == 1
                tensors = tuple(t.squeeze(0) for t in tensors)
            return tensors if is_seq else tensors[0]

        if isinstance(self.backbone, nn.Module):
            x = x / 255.0
            lead_dim, T, B, x_shape = infer_leading_dims(x, 3)

            backbone_out = self.backbone(
                x.view(T * B, *x_shape)
            )  # Fold if T dimension.
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            lead_dim, T, B, x_shape = infer_leading_dims(x, 1)
            backbone_out = x
        lstm_input = self.fc(backbone_out)
        lstm_input = torch.cat(
            [
                lstm_input.view(T, B, -1),
                prev_action.view(T, B, -1),
                prev_reward.view(T, B, 1),
            ],
            dim=2,
        )
        hidden = torch.transpose(hidden, 0, 1)
        hidden = None if hidden is None else hidden
        lstm_out, hidden = self.gru(lstm_input, hidden)

        q = self.head(lstm_out.contiguous().view(T * B, -1))

        # Restore leading dimensions: [T,B], [B], or [], as input.
        q = restore_leading_dims(q, lead_dim, T, B)

        return q, hidden

    def forward_(
        self, x: torch.Tensor, hidden: torch.Tensor, n_tau_samples: int = None
    ):
        def infer_leading_dims(tensor, dim):
            """Looks for up to two leading dimensions in ``tensor``, before
            the data dimensions, of which there are assumed to be ``dim`` number.
            For use at beginning of model's ``forward()`` method, which should
            finish with ``restore_leading_dims()`` (see that function for help.)
            Returns:
            lead_dim: int --number of leading dims found.
            T: int --size of first leading dim, if two leading dims, o/w 1.
            B: int --size of first leading dim if one, second leading dim if two, o/w 1.
            shape: tensor shape after leading dims.
            """
            lead_dim = tensor.dim() - dim
            assert lead_dim in (0, 1, 2)
            if lead_dim == 2:
                T, B = tensor.shape[:2]
            else:
                T = 1
                B = 1 if lead_dim == 0 else tensor.shape[0]
            shape = tensor.shape[lead_dim:]
            return lead_dim, T, B, shape

        def restore_leading_dims(tensors, lead_dim, T=1, B=1):
            """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
            leading dimensions, which will become [], [B], or [T,B].  Assumes input
            tensors already have a leading Batch dimension, which might need to be
            removed. (Typically the last layer of model will compute with leading
            batch dimension.)  For use in model ``forward()`` method, so that output
            dimensions match input dimensions, and the same model can be used for any
            such case.  Use with outputs from ``infer_leading_dims()``."""
            is_seq = isinstance(tensors, (tuple, list))
            tensors = tensors if is_seq else (tensors,)
            if lead_dim == 2:  # (Put T dim.)
                tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
            if lead_dim == 0:  # (Remove B=1 dim.)
                assert B == 1
                tensors = tuple(t.squeeze(0) for t in tensors)
            return tensors if is_seq else tensors[0]

        if isinstance(self.backbone, nn.Module):
            img = x / 255.0
            lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

            conv_out = self.backbone(
                img.view(T * B, *img_shape)
            )  # Fold if T dimension.

            lstm_input = conv_out.view(T, B, -1)
            hidden = torch.transpose(hidden, 0, 1)
            hidden = None if hidden is None else hidden
            lstm_out, hidden = self.gru(lstm_input, hidden)

            q = self.head(lstm_out.view(T * B, -1))

            # Restore leading dimensions: [T,B], [B], or [], as input.
            q = restore_leading_dims(q, lead_dim, T, B)

            return q, hidden
        else:
            if len(x.shape) == 1:
                x = x.reshape(1, 1, -1)
            hidden = torch.transpose(hidden, 0, 1)
            x, hidden = self.gru(x, hidden)
            x = self.head(x)
            return x, hidden

    def calculate_fc_input_size(self, state_dim: tuple):
        """Calculate fc input size according to the shape of cnn."""
        x = torch.zeros(state_dim).unsqueeze(0)
        output = self.backbone(x).detach().view(-1)
        return output.shape[0]
