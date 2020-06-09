# -*- coding: utf-8 -*-
"""R2D1 agent, which implement R2D2 but without distributed actor.
- Author: Kyunghwan Kim, Curt Park, Euijin Jeong
- Contact:kh.kim@medipixel.io, curt.park@medipixel.io, euijin.jeong@medipixel.io
- Paper: https://openreview.net/pdf?id=r1lyTjAqYX (R2D1)
"""

import time
from typing import Tuple

import numpy as np
import torch
import wandb

from rl_algorithms.common.buffer.recurrent_replay_buffer import RecurrentReplayBuffer
import rl_algorithms.common.helper_functions as common_utils
from rl_algorithms.common.wrapper import PERWrapper
from rl_algorithms.dqn.agent import DQNAgent
from rl_algorithms.dqn.learner import R2D1Learner
from rl_algorithms.registry import AGENTS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@AGENTS.register_module
class R2D1Agent(DQNAgent):
    """R2D1 interacting with environment.

    Attribute:
        memory (RecurrentPrioritizedReplayBuffer): replay memory for recurrent agent
        memory_n (RecurrentReplayBuffer): nstep replay memory for recurrent agent
    """

    # pylint: disable=attribute-defined-outside-init
    def _initialize(self):
        """Initialize non-common things."""
        if not self.args.test:

            self.memory = RecurrentReplayBuffer(
                self.hyper_params.buffer_size,
                self.hyper_params.batch_size,
                self.hyper_params.sequence_size,
                self.hyper_params.overlap_size,
            )
            self.memory = PERWrapper(self.memory, alpha=self.hyper_params.per_alpha)

            # replay memory for multi-steps
            if self.use_n_step:
                self.memory_n = RecurrentReplayBuffer(
                    self.hyper_params.buffer_size,
                    self.hyper_params.batch_size,
                    self.hyper_params.sequence_size,
                    self.hyper_params.overlap_size,
                    n_step=self.hyper_params.n_step,
                    gamma=self.hyper_params.gamma,
                )

        self.learner = R2D1Learner(
            self.args,
            self.hyper_params,
            self.log_cfg,
            self.head_cfg,
            self.backbone_cfg,
            self.optim_cfg,
            device,
        )

    def select_action(
        self,
        state: np.ndarray,
        hidden_state: torch.Tensor,
        prev_action: torch.Tensor,
        prev_reward: np.ndarray,
    ) -> np.ndarray:
        """Select an action from the input space."""
        self.curr_state = state

        # epsilon greedy policy
        # pylint: disable=comparison-with-callable
        state = self._preprocess_state(state)
        selected_action, hidden_state = self.learner.dqn(
            state, hidden_state, prev_action, prev_reward
        )
        selected_action = selected_action.detach().argmax().cpu().numpy()
        if not self.args.test and self.epsilon > np.random.random():
            selected_action = np.array(self.env.action_space.sample())
        return selected_action, hidden_state

    def step(
        self, action: np.ndarray, hidden_state: torch.Tensor
    ) -> Tuple[np.ndarray, np.float64, bool, dict]:
        """Take an action and return the response of the env."""
        next_state, reward, done, info = self.env.step(action)
        if not self.args.test:
            # if the last state is not a terminal state, store done as false
            done_bool = (
                False if self.episode_step == self.args.max_episode_steps else done
            )

            transition = (
                self.curr_state,
                action,
                hidden_state.detach(),
                reward,
                next_state,
                done_bool,
            )
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def train(self):
        """Train the agent."""
        # logger
        if self.args.log:
            self.set_wandb()
            # wandb.watch([self.dqn], log="parameters")

        # pre-training if needed
        self.pretrain()

        for self.i_episode in range(1, self.args.episode_num + 1):
            state = self.env.reset()
            self.episode_step = 0
            self.sequence_step = 0
            losses = list()
            done = False
            score = 0

            hidden_in = torch.zeros(
                [1, 1, self.head_cfg.configs.rnn_hidden_size], dtype=torch.float
            ).to(device)
            prev_action = torch.zeros(1, 1, self.head_cfg.configs.output_size).to(
                device
            )
            prev_reward = torch.zeros(1, 1, 1).to(device)

            t_begin = time.time()

            while not done:
                if self.args.render and self.i_episode >= self.args.render_after:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)
                self.total_step += 1
                self.episode_step += 1

                if self.episode_step % self.hyper_params.sequence_size == 0:
                    self.sequence_step += 1

                if len(self.memory) >= self.hyper_params.update_starts_from:
                    if self.sequence_step % self.hyper_params.train_freq == 0:
                        for _ in range(self.hyper_params.multiple_update):
                            experience = self.sample_experience()
                            info = self.learner.update_model(experience)
                            loss = info[0:2]
                            indices, new_priorities = info[2:4]
                            losses.append(loss)  # for logging
                            self.memory.update_priorities(indices, new_priorities)

                    # decrease epsilon
                    self.epsilon = max(
                        self.epsilon
                        - (self.max_epsilon - self.min_epsilon)
                        * self.hyper_params.epsilon_decay,
                        self.min_epsilon,
                    )

                    # increase priority beta
                    fraction = min(float(self.i_episode) / self.args.episode_num, 1.0)
                    self.per_beta = self.per_beta + fraction * (1.0 - self.per_beta)

                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.head_cfg.configs.output_size
                )
                prev_reward = torch.as_tensor(reward).to(device)
                score += reward

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)

                if self.i_episode % self.args.save_period == 0:
                    self.learner.save_params(self.i_episode)
                    self.interim_test()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()

    def _test(self, interim_test: bool = False):
        """Common test routine."""

        if interim_test:
            test_num = self.args.interim_test_num
        else:
            test_num = self.args.episode_num
        score_list = []
        for i_episode in range(test_num):
            hidden_in = torch.zeros(
                [1, 1, self.head_cfg.configs.rnn_hidden_size], dtype=torch.float
            ).to(device)
            prev_action = torch.zeros(1, 1, self.head_cfg.configs.output_size).to(
                device
            )
            prev_reward = torch.zeros(1, 1, 1).to(device)
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.args.render:
                    self.env.render()

                action, hidden_out = self.select_action(
                    state, hidden_in, prev_action, prev_reward
                )
                next_state, reward, done, _ = self.step(action, hidden_in)

                hidden_in = hidden_out
                state = next_state
                prev_action = common_utils.make_one_hot(
                    torch.as_tensor(action), self.head_cfg.configs.output_size
                )
                prev_reward = torch.as_tensor(reward).to(device)
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            score_list.append(score)

        if self.args.log:
            wandb.log(
                {
                    "test score": round(sum(score_list) / len(score_list), 2),
                    "test total step": self.total_step,
                }
            )
