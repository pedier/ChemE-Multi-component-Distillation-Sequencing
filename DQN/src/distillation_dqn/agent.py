"""
Last modified time: 2026-09-21-02:14
Last modified content: Implement the masked Dueling Double DQN agent core
Last modified by: OpenAI Codex
File design: Validated learner configuration and CPU agent operations
File purpose: Select legal towers, replay experiences, and update action values
File creator: OpenAI Codex
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn

from distillation_sequencing_env.environment import action_mask, available_actions
from distillation_dqn.models import ActionMask, State
from distillation_dqn.network import DuelingQNetwork
from distillation_dqn.replay import ReplayBuffer, Transition
from distillation_dqn.targets import double_dqn_targets


_TIE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DQNConfig:
    """Hold validated fixed-instance DQN learning parameters.

    Inputs:
        learning_rate: Positive Adam learning rate.
        discount_factor: Economic discount factor, fixed at one.
        replay_capacity: Maximum retained transitions.
        batch_size: Number of sampled transitions per update.
        warmup_steps: Minimum replay size before updating.
        target_sync_interval: Optimization steps between target copies.
        gradient_clip_norm: Positive maximum gradient norm.
        seed: Integer seed for independent exploration and initialization.
    Returns:
        Immutable configuration for the DQN agent.
    """

    learning_rate: float = 1e-3
    discount_factor: float = 1.0
    replay_capacity: int = 10_000
    batch_size: int = 64
    warmup_steps: int = 128
    target_sync_interval: int = 100
    gradient_clip_norm: float = 10.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate numerical and discrete learning parameters.

        Inputs:
            self: Newly constructed configuration.
        Returns:
            None. Raises ValueError or TypeError for invalid fields.
        """

        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if self.discount_factor != 1.0:
            raise ValueError("discount_factor must equal 1.0 for annual cost.")
        if not math.isfinite(self.gradient_clip_norm) or self.gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be positive and finite.")

        # These bounds guarantee that every optimization sample fits in replay.
        integer_fields = (self.replay_capacity, self.batch_size, self.warmup_steps)
        integer_fields += (self.target_sync_interval, self.seed)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields):
            raise TypeError("capacity, batch, warmup, sync interval, and seed must be integers.")
        if not 0 < self.batch_size <= self.warmup_steps <= self.replay_capacity:
            raise ValueError("require 0 < batch_size <= warmup_steps <= replay_capacity.")
        if self.target_sync_interval <= 0:
            raise ValueError("target_sync_interval must be positive.")


class DuelingDoubleDQNAgent:
    """Learn masked tower values with online and frozen target networks.

    Inputs:
        config: Optional validated DQN configuration.
    Returns:
        CPU agent with independent exploration and replay generators.
    """

    def __init__(self, config: DQNConfig | None = None) -> None:
        """Create seeded networks, Adam optimizer, and replay buffer.

        Inputs:
            config: Optional validated DQN configuration.
        Returns:
            None.
        """

        self.config = config or DQNConfig()
        self._random = random.Random(self.config.seed)
        self.replay = ReplayBuffer(self.config.replay_capacity, self.config.seed + 1)
        self.optimization_steps = 0

        # Forking protects unrelated PyTorch users from this agent's seed.
        torch.use_deterministic_algorithms(True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.seed)
            self.online_network = DuelingQNetwork().cpu()
            self.target_network = DuelingQNetwork().cpu()
        self.sync_target()
        self.target_network.requires_grad_(False)
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=self.config.learning_rate
        )

    def q_values(self, state: State) -> dict[int, float]:
        """Read network estimates only for feasible tower actions.

        Inputs:
            state: Six-entry nonterminal or terminal distillation state.
        Returns:
            Mapping of legal textbook tower IDs to Q values.
        """

        actions = available_actions(state)
        if not actions:
            return {}

        states = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            values = self.online_network(states).squeeze(0).tolist()

        return {action_id: values[action_id - 1] for action_id in actions}

    def greedy_action(self, state: State) -> int:
        """Choose the highest-valued legal tower with stable tie handling.

        Inputs:
            state: Nonterminal six-entry distillation state.
        Returns:
            Smallest feasible tower ID within the absolute tie tolerance.
        """

        values = self.q_values(state)
        if not values:
            raise ValueError("A greedy action cannot be selected in a terminal state.")

        best_value = max(values.values())
        ties = [
            action_id
            for action_id, value in values.items()
            if math.isclose(value, best_value, rel_tol=0.0, abs_tol=_TIE_ABS_TOLERANCE)
        ]
        return min(ties)

    def select_action(self, state: State, mask: ActionMask, epsilon: float) -> int:
        """Use masked epsilon-greedy exploration without illegal towers.

        Inputs:
            state: Current nonterminal distillation state.
            mask: Environment-provided action mask for that state.
            epsilon: Probability of a uniform legal exploratory action.
        Returns:
            One feasible textbook tower ID.
        """

        if mask != action_mask(state):
            raise ValueError("mask does not match the current state.")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")

        actions = available_actions(state)
        if not actions:
            raise ValueError("An action cannot be selected in a terminal state.")
        if self._random.random() < epsilon:
            return self._random.choice(actions)
        return self.greedy_action(state)

    def sync_target(self) -> None:
        """Copy online weights to the target network exactly.

        Inputs:
            None.
        Returns:
            None. Updates target weights without gradients.
        """

        self.target_network.load_state_dict(self.online_network.state_dict())

    def optimize(self) -> float | None:
        """Run one Huber-loss Double DQN update after replay warmup.

        Inputs:
            None. Samples the agent's seeded replay buffer.
        Returns:
            Scalar loss, or None before enough transitions have accumulated.
        """

        if len(self.replay) < self.config.warmup_steps:
            return None

        transitions = self.replay.sample(self.config.batch_size)
        states = torch.tensor([item.state for item in transitions], dtype=torch.float32)
        indices = torch.tensor([item.action_id - 1 for item in transitions]).unsqueeze(1)
        predictions = self.online_network(states).gather(1, indices).squeeze(1)
        targets = double_dqn_targets(self.online_network, self.target_network, transitions)
        loss = nn.functional.smooth_l1_loss(predictions, targets)

        # Update only the online network; synchronize the frozen target periodically.
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_sync_interval == 0:
            self.sync_target()

        return float(loss.item())

    def observe(self, transition: Transition) -> float | None:
        """Store an observed transition and attempt one optimization.

        Inputs:
            transition: Validated legal environment transition.
        Returns:
            Scalar loss if warmup is complete, otherwise None.
        """

        self.replay.push(transition)
        return self.optimize()

