"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Sparse legal-action Q table with deterministic policy extraction
File purpose: Provide masked epsilon-greedy selection and Bellman updates
File creator: OpenAI Codex
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass

from distillation_sequencing_env.environment import available_actions
from distillation_q_learning.models import State


_TIE_ABS_TOLERANCE = 1e-12


@dataclass(frozen=True)
class QLearningConfig:
    """Define validated hyperparameters for tabular Q-learning.

    Inputs:
        learning_rate: Fraction of each temporal-difference error to apply.
        discount_factor: Economic discount factor, fixed at one for this problem.
        epsilon_start: Exploration probability at episode zero.
        epsilon_min: Lower bound for exploration probability.
        epsilon_decay: Multiplicative exploration decay per episode.
        seed: Seed for the agent's independent random generator.
    Returns:
        An immutable, validated configuration.
    """

    learning_rate: float = 0.2
    discount_factor: float = 1.0
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate every Q-learning hyperparameter.

        Inputs:
            self: Newly constructed configuration.
        Returns:
            None. Raises ValueError or TypeError for invalid values.
        """

        # Validation preserves a stable learning update and the undiscounted economic objective.
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1].")
        if self.discount_factor != 1.0:
            raise ValueError("discount_factor must equal 1.0 for the annual-cost objective.")
        if not 0.0 <= self.epsilon_min <= self.epsilon_start <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= minimum <= start <= 1.")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be in (0, 1].")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer.")


def epsilon_for_episode(config: QLearningConfig, episode: int) -> float:
    """Calculate the bounded exploration rate for one episode.

    Inputs:
        config: Validated Q-learning hyperparameters.
        episode: Zero-based episode index.
    Returns:
        Exploration probability after exponential decay and lower-bound clipping.
    """

    if episode < 0:
        raise ValueError("episode must be non-negative.")

    return max(config.epsilon_min, config.epsilon_start * config.epsilon_decay**episode)


class TabularQLearningAgent:
    """Learn action values for legal distillation state-action pairs.

    Inputs:
        config: Validated learning and exploration parameters.
    Returns:
        A sparse Q-table agent with an independent random generator.
    """

    def __init__(self, config: QLearningConfig | None = None) -> None:
        """Initialize an empty Q table and seeded random generator.

        Inputs:
            config: Optional configuration; defaults are used when omitted.
        Returns:
            None.
        """

        self.config = config or QLearningConfig()
        self._q_values: dict[tuple[State, int], float] = {}
        self._random = random.Random(self.config.seed)

    def q_value(self, state: State, action_id: int) -> float:
        """Read one legal action value, using zero for an unvisited pair.

        Inputs:
            state: Six-element distillation state.
            action_id: Feasible textbook column ID.
        Returns:
            Stored Q value or zero when the legal pair has not been visited.
        """

        if type(action_id) is not int or action_id not in available_actions(state):
            raise ValueError(f"Action {action_id} is infeasible in the supplied state.")

        return self._q_values.get((state, action_id), 0.0)

    def q_values(self, state: State) -> dict[int, float]:
        """Return a copy of all legal action values for one state.

        Inputs:
            state: Six-element distillation state.
        Returns:
            Mapping from each feasible action ID to its current Q value.
        """

        return {action_id: self._q_values.get((state, action_id), 0.0)
                for action_id in available_actions(state)}

    def greedy_action(self, state: State) -> int:
        """Select the highest-valued legal action with deterministic tie handling.

        Inputs:
            state: Nonterminal six-element distillation state.
        Returns:
            Smallest action ID whose Q value is within tolerance of the maximum.
        """

        values = self.q_values(state)
        if not values:
            raise ValueError("A greedy action cannot be selected in a terminal state.")

        best_value = max(values.values())
        tied_actions = [
            action_id
            for action_id, value in values.items()
            if math.isclose(value, best_value, rel_tol=0.0, abs_tol=_TIE_ABS_TOLERANCE)
        ]
        return min(tied_actions)

    def select_action(self, state: State, epsilon: float) -> int:
        """Select a legal action with masked epsilon-greedy exploration.

        Inputs:
            state: Nonterminal six-element distillation state.
            epsilon: Probability of uniformly exploring legal actions.
        Returns:
            Selected feasible textbook action ID.
        """

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")

        actions = available_actions(state)
        if not actions:
            raise ValueError("An action cannot be selected in a terminal state.")

        if self._random.random() < epsilon:
            return self._random.choice(actions)

        return self.greedy_action(state)

    def update(
        self,
        state: State,
        action_id: int,
        reward: float,
        next_state: State,
        terminated: bool,
    ) -> float:
        """Apply one masked tabular Q-learning update.

        Inputs:
            state: State before the observed transition.
            action_id: Feasible action taken in the state.
            reward: Immediate negative annual-cost reward.
            next_state: State returned by the environment.
            terminated: Whether the transition completed the flowsheet.
        Returns:
            Updated Q value for the observed state-action pair.
        """

        current_value = self.q_value(state, action_id)
        next_actions = available_actions(next_state)
        if terminated and next_actions:
            raise ValueError("A terminal transition cannot have feasible next actions.")
        if not terminated and not next_actions:
            raise ValueError("A nonterminal transition must have a feasible next action.")

        # Only feasible next actions contribute to the Bellman target.
        future_value = 0.0
        if not terminated:
            future_value = max(self._q_values.get((next_state, action), 0.0) for action in next_actions)

        target = reward + self.config.discount_factor * future_value
        updated_value = current_value + self.config.learning_rate * (target - current_value)
        self._q_values[(state, action_id)] = updated_value
        return updated_value

    def greedy_policy(self, states: Iterable[State]) -> dict[State, int]:
        """Extract a deterministic pure policy for supplied nonterminal states.

        Inputs:
            states: Iterable of nonterminal distillation states.
        Returns:
            Mapping from each state to its deterministic greedy action.
        """

        return {state: self.greedy_action(state) for state in states}

