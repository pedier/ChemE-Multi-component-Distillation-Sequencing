"""
Last modified time: 2026-09-24
Last modified content: Consolidate verified chemistry and remove redundant work
Last modified by: OpenAI Codex
File design: Immutable transition records and bounded replay storage
File purpose: Reuse genuine legal distillation transitions for off-policy learning
File creator: OpenAI Codex
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from distillation_sequencing_env.data import TERMINAL_STATE
from distillation_sequencing_env.environment import action_mask, apply_action, validate_reward
from distillation_dqn.models import ActionMask, State


@dataclass(frozen=True)
class Transition:
    """Store one physically valid observed distillation transition.

    Inputs:
        state: State before selecting a tower.
        action_id: Selected textbook tower ID.
        reward: Observed negative annualized cost in M$/yr.
        next_state: State after the sharp split.
        next_action_mask: Feasible towers after the split.
        terminated: Whether all products are pure.
    Returns:
        Immutable transition record for replay.
    """

    state: State
    action_id: int
    reward: float
    next_state: State
    next_action_mask: ActionMask
    terminated: bool

    def __post_init__(self) -> None:
        """Reject records that contradict the deterministic environment.

        Inputs:
            self: Newly constructed transition.
        Returns:
            None. Raises ValueError for an inconsistent transition.
        """

        if apply_action(self.state, self.action_id) != self.next_state:
            raise ValueError("next_state does not match the selected tower.")
        if self.next_action_mask != action_mask(self.next_state):
            raise ValueError("next_action_mask does not match next_state.")
        if self.terminated != (self.next_state == TERMINAL_STATE):
            raise ValueError("terminated does not match next_state.")
        validate_reward(self.action_id, self.reward)


class ReplayBuffer:
    """Sample past legal transitions uniformly with independent randomness.

    Inputs:
        capacity: Maximum retained transition count.
        seed: Seed for an independent Python random generator.
    Returns:
        Bounded replay storage with reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        """Initialize a bounded transition ring and random generator.

        Inputs:
            capacity: Positive maximum retained transition count.
            seed: Integer random seed.
        Returns:
            None.
        """

        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        self.capacity = capacity
        self._transitions: list[Transition] = []
        self._oldest = 0
        self._random = random.Random(seed)

    def __len__(self) -> int:
        """Return the number of retained transitions.

        Inputs:
            None.
        Returns:
            Current replay-buffer length.
        """

        return len(self._transitions)

    def push(self, transition: Transition) -> None:
        """Append one transition, evicting the oldest at capacity.

        Inputs:
            transition: Validated immutable transition.
        Returns:
            None.
        """

        if not isinstance(transition, Transition):
            raise TypeError("transition must be a Transition record.")
        if len(self) < self.capacity:
            self._transitions.append(transition)
        else:
            self._transitions[self._oldest] = transition
            self._oldest = (self._oldest + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple[Transition, ...]:
        """Uniformly sample distinct retained transitions.

        Inputs:
            batch_size: Positive sample size no larger than current length.
        Returns:
            Reproducible tuple sampled without replacement.
        """

        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be positive and fit the replay buffer.")

        indices = self._random.sample(range(len(self)), batch_size)
        return tuple(self._transitions[(self._oldest + index) % len(self)] for index in indices)
