import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.network import DQN
from agents.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        gamma=0.99,
        learning_rate=0.001,
        batch_size=64,
        replay_memory_size=100_000,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        target_update_frequency=10
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.batch_size = batch_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.target_update_frequency = (
            target_update_frequency
        )

        # Device
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Policy network
        self.policy_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Target network
        self.target_network = DQN(
            state_size,
            action_size
        ).to(self.device)

        # Initially both networks are identical
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_network.parameters(),
            lr=learning_rate
        )

        # Loss
        self.loss_function = nn.MSELoss()

        # Replay buffer
        self.memory = ReplayBuffer(
            replay_memory_size
        )

        self.training_steps = 0

    def select_action(self, state, training=True):
        """
        Epsilon-greedy action selection.
        """

        # Exploration
        if training and random.random() < self.epsilon:
            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_network(
                state_tensor
            )

        action = torch.argmax(
            q_values,
            dim=1
        ).item()

        return action

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done
        )

    def train_step(self):
        """
        Perform one DQN training step.
        """

        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(
            self.batch_size
        )

        states = np.array([
            experience[0]
            for experience in batch
        ])

        actions = np.array([
            experience[1]
            for experience in batch
        ])

        rewards = np.array([
            experience[2]
            for experience in batch
        ])

        next_states = np.array([
            experience[3]
            for experience in batch
        ])

        dones = np.array([
            experience[4]
            for experience in batch
        ])

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        )

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        # -----------------------------------
        # Current Q values
        # -----------------------------------

        current_q_values = self.policy_network(
            states
        )

        current_q_values = current_q_values.gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        # -----------------------------------
        # Target Q values
        # -----------------------------------

        with torch.no_grad():

            next_q_values = self.target_network(
                next_states
            )

            max_next_q_values = next_q_values.max(
                dim=1
            ).values

            target_q_values = (
                rewards
                + self.gamma
                * max_next_q_values
                * (1 - dones)
            )

        # -----------------------------------
        # Loss
        # -----------------------------------

        loss = self.loss_function(
            current_q_values,
            target_q_values
        )

        # -----------------------------------
        # Backpropagation
        # -----------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_network.parameters(),
            max_norm=1.0
        )

        self.optimizer.step()

        self.training_steps += 1

        # -----------------------------------
        # Epsilon decay
        # -----------------------------------

        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay

            self.epsilon = max(
                self.epsilon,
                self.epsilon_min
            )

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )

    def save(self, path):
        torch.save(
            self.policy_network.state_dict(),
            path
        )

    def load(self, path):
        self.policy_network.load_state_dict(
            torch.load(
                path,
                map_location=self.device
            )
        )

        self.target_network.load_state_dict(
            self.policy_network.state_dict()
        )