# -*- coding: utf-8 -*-
"""Replay buffer for baselines."""

from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import torch

from rl_algorithms.common.helper_functions import get_n_step_info

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RecurrentReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    Attributes:
        buffer (list): list of replay buffer
        batch_size (int): size of a batched sampled from replay buffer for training
    """

    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
        sequence_size: int,
        overlap_size: int,
        gamma: float = 0.99,
        n_step: int = 1,
        demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
    ):
        """Initialize a ReplayBuffer object.
        Args:
            buffer_size (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
        """
        self.buffer: list = list()
        self.local_buffer: list = list()
        self.n_step = n_step
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.overlap_size = overlap_size
        self.sequence_size = sequence_size
        self.episode_idx = 0
        self.idx = 0
        self.demo = demo
        self.n_step_buffer: Deque = deque(maxlen=n_step)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        hidden: torch.Tensor,
        reward: np.float64,
        next_state: np.ndarray,
        done: float,
    ):
        """Add a new experience to memory."""
        data = [state, action, hidden, reward, next_state, done]

        self.n_step_buffer.append(data)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # add a multi step transition
        reward, _, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action, hidden_state = self.n_step_buffer[0][:3]

        self.local_buffer.append([curr_state, action, hidden_state, reward, done])
        self.idx += 1

        if done and self.idx < self.sequence_size:
            while self.idx < self.sequence_size:
                self.local_buffer.append(
                    [
                        np.zeros(curr_state.shape),
                        np.array(0),
                        torch.zeros(hidden_state.shape).to(device),
                        0.0,
                        True,
                    ]
                )
                self.idx += 1

        if self.idx % self.sequence_size == 0:
            self.buffer.append(self.local_buffer)
            self.idx = self.overlap_size
            self.episode_idx += 1
            self.local_buffer = self.local_buffer[self.overlap_size :]
            self.episode_idx = (
                0 if self.episode_idx % self.buffer_size == 0 else self.episode_idx
            )

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(self, transitions: list):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(*transition)

    def sample(self, indices: List[int] = None) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states, actions, hiddens, rewards, dones = [], [], [], [], []

        for i in indices:
            s, a, h, r, d = map(list, zip(*self.buffer[i]))
            states.append(np.array(s, copy=False))
            actions.append(np.array(a, copy=False))
            hiddens.append(torch.stack(h))
            rewards.append(np.array(r, copy=False))
            dones.append(np.array(d, copy=False, dtype=np.uint8))

        states_ = torch.FloatTensor(np.stack(states)).to(device)
        actions_ = torch.FloatTensor(np.stack(actions)).to(device)
        hiddens_ = torch.stack(hiddens)
        rewards_ = torch.FloatTensor(np.stack(rewards)).to(device)
        dones_ = torch.FloatTensor(np.array(dones)).to(device)

        if torch.cuda.is_available():
            states_ = states_.cuda(non_blocking=True)
            actions_ = actions_.cuda(non_blocking=True)
            hiddens_ = hiddens_.cuda(non_blocking=True)
            rewards_ = rewards_.cuda(non_blocking=True)
            dones_ = dones_.cuda(non_blocking=True)

        return [states_, actions_, rewards_, hiddens_, dones_, 0]

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
