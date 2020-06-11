from typing import Any, Tuple

import torch

from rl_algorithms.common.helper_functions import make_one_hot
from rl_algorithms.utils.config import ConfigDict


def infer_leading_dims(tensor: torch.Tensor, dim: int) -> Tuple[int, int, int, Tuple]:
    """Looks for up to two leading dimensions in ``tensor``, before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.

    Cloned from rlpyt repo:
    https://github.com/astooke/rlpyt/blob/master/rlpyt/models/dqn/atari_r2d1_model.py
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


def restore_leading_dims(
    tensors: torch.Tensor, lead_dim: int, T: int = 1, B: int = 1
) -> torch.Tensor:
    """Reshapes ``tensors`` (one or `tuple`, `list`) to to have ``lead_dim``
    leading dimensions, which will become [], [B], or [T,B].  Assumes input
    tensors already have a leading Batch dimension, which might need to be
    removed. (Typically the last layer of model will compute with leading
    batch dimension.)  For use in model ``forward()`` method, so that output
    dimensions match input dimensions, and the same model can be used for any
    such case.  Use with outputs from ``infer_leading_dims()``.

    Cloned from rlpyt repo:
    https://github.com/astooke/rlpyt/blob/master/rlpyt/models/dqn/atari_r2d1_model.py
    """
    is_seq = isinstance(tensors, (tuple, list))
    tensors = tensors if is_seq else (tensors,)
    if lead_dim == 2:  # (Put T dim.)
        tensors = tuple(t.view((T, B) + t.shape[1:]) for t in tensors)
    if lead_dim == 0:  # (Remove B=1 dim.)
        assert B == 1
        tensors = tuple(t.squeeze(0) for t in tensors)
    return tensors if is_seq else tensors[0]


def valid_from_done(done: torch.Tensor) -> torch.Tensor:
    """Returns a float mask which is zero for all time-steps after a
    `done=True` is signaled.  This function operates on the leading dimension
    of `done`, assumed to correspond to time [T,...], other dimensions are
    preserved.

    Cloned from rlpyt repo:
        https://github.com/astooke/rlpyt/blob/master/rlpyt/algos/utils.py
    """
    done = done.type(torch.float).squeeze()
    valid = torch.ones_like(done)
    valid[1:] = 1 - torch.clamp(torch.cumsum(done[:-1], dim=0), max=1)
    valid = valid[-1] == 0
    return valid


def slice_r2d1_arguments(experiences: Tuple[Any, ...], head_cfg: ConfigDict) -> Tuple:
    states, actions, rewards, hiddens, dones = experiences[:5]

    burnin_states = states[:, 1 : head_cfg.configs.burn_in_step]
    target_burnin_states = states[:, 1 : head_cfg.configs.burn_in_step + 1]
    agent_states = states[:, head_cfg.configs.burn_in_step : -1]
    target_states = states[:, head_cfg.configs.burn_in_step + 1 :]

    burnin_prev_actions = make_one_hot(
        actions[:, : head_cfg.configs.burn_in_step - 1], head_cfg.configs.output_size,
    )
    target_burnin_prev_actions = make_one_hot(
        actions[:, : head_cfg.configs.burn_in_step], head_cfg.configs.output_size
    )
    agent_actions = actions[:, head_cfg.configs.burn_in_step : -1].long().unsqueeze(-1)
    prev_actions = make_one_hot(
        actions[:, head_cfg.configs.burn_in_step - 1 : -2],
        head_cfg.configs.output_size,
    )
    target_prev_actions = make_one_hot(
        actions[:, head_cfg.configs.burn_in_step : -1].long(),
        head_cfg.configs.output_size,
    )

    burnin_prev_rewards = rewards[:, : head_cfg.configs.burn_in_step - 1].unsqueeze(-1)
    target_burnin_prev_rewards = rewards[:, : head_cfg.configs.burn_in_step].unsqueeze(
        -1
    )
    agent_rewards = rewards[:, head_cfg.configs.burn_in_step : -1].unsqueeze(-1)
    prev_rewards = rewards[:, head_cfg.configs.burn_in_step - 1 : -2].unsqueeze(-1)
    target_prev_rewards = agent_rewards
    burnin_dones = dones[:, 1 : head_cfg.configs.burn_in_step].unsqueeze(-1)
    burnin_target_dones = dones[:, 1 : head_cfg.configs.burn_in_step + 1].unsqueeze(-1)
    agent_dones = dones[:, head_cfg.configs.burn_in_step : -1].unsqueeze(-1)
    init_rnn_state = hiddens[:, 0].squeeze(1).contiguous()

    burnin_state_tuple = (burnin_states, target_burnin_states)
    state_tuple = (agent_states, target_states)
    burnin_prev_action_tuple = (burnin_prev_actions, target_burnin_prev_actions)
    prev_action_tuple = (prev_actions, target_prev_actions)
    burnin_prev_reward_tuple = (burnin_prev_rewards, target_burnin_prev_rewards)
    prev_reward_tuple = (prev_rewards, target_prev_rewards)
    burnin_dones_tuple = (burnin_dones, burnin_target_dones)

    return (
        burnin_state_tuple,
        state_tuple,
        burnin_prev_action_tuple,
        agent_actions,
        prev_action_tuple,
        burnin_prev_reward_tuple,
        agent_rewards,
        prev_reward_tuple,
        burnin_dones_tuple,
        agent_dones,
        init_rnn_state,
    )
