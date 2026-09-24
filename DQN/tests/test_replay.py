"""
Last modified time: 2026-09-24
Last modified content: Verify shared contracts, strict inputs and seeded cleanup behavior
Last modified by: OpenAI Codex
File design: Replay record and buffer unit tests
File purpose: Prevent impossible chemical transitions from entering DQN replay
File creator: OpenAI Codex
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest

from distillation_dqn import DistillationSequenceEnvironment, ReplayBuffer, Transition


@pytest.fixture
def observed_transition() -> Transition:
    """Create one transition from a genuine legal environment step.

    Inputs:
        None.
    Returns:
        The initial tower-two transition with next-state mask.
    """

    environment = DistillationSequenceEnvironment()
    state, _ = environment.reset()
    result = environment.step(2)
    return Transition(state, 2, result.reward, result.state, result.action_mask, result.terminated)


def test_transition_is_immutable_and_matches_environment(observed_transition: Transition) -> None:
    """Accept a physical step and retain its complete learning record.

    Inputs:
        observed_transition: Actual first split using tower two.
    Returns:
        None. The test passes when fields and immutability are correct.
    """

    assert observed_transition.action_id == 2
    assert observed_transition.reward == pytest.approx(-1.654600)
    assert observed_transition.next_action_mask[7]
    assert observed_transition.next_action_mask[9]
    assert not observed_transition.terminated
    with pytest.raises(FrozenInstanceError):
        observed_transition.reward = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"action_id": 8}, "不可行"),
        ({"next_state": (0, 0, 0, 0, 0, 0)}, "next_state"),
        ({"next_action_mask": (False,) * 10}, "next_action_mask"),
        ({"terminated": True}, "terminated"),
        ({"reward": float("nan")}, "finite"),
        ({"reward": 0.0}, "cost"),
    ],
)
def test_transition_rejects_inconsistent_fields(
    observed_transition: Transition, changes: dict[str, object], message: str
) -> None:
    """Reject illegal towers, false transitions, masks, flags, or rewards.

    Inputs:
        observed_transition: Valid baseline record.
        changes: One field replacement that breaks physical consistency.
        message: Expected validation fragment.
    Returns:
        None. The test passes when construction fails.
    """

    with pytest.raises(ValueError, match=message):
        replace(observed_transition, **changes)


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_replay_rejects_invalid_capacity(capacity: object) -> None:
    """Require a positive integer replay capacity.

    Inputs:
        capacity: Invalid capacity value.
    Returns:
        None. The test passes when construction rejects it.
    """

    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=capacity)  # type: ignore[arg-type]


@pytest.mark.parametrize("seed", [1.5, True])
def test_replay_rejects_invalid_seed(seed: object) -> None:
    """Require an integer replay random seed.

    Inputs:
        seed: Invalid seed value.
    Returns:
        None. The test passes when construction raises TypeError.
    """

    with pytest.raises(TypeError, match="seed"):
        ReplayBuffer(seed=seed)  # type: ignore[arg-type]


def test_replay_evicts_oldest_and_samples_reproducibly(observed_transition: Transition) -> None:
    """Check capacity eviction, length, and independent seeded samples.

    Inputs:
        observed_transition: Initial legal transition used for three records.
    Returns:
        None. The test passes when oldest data leaves and samples match.
    """

    records = []
    environment = DistillationSequenceEnvironment()
    state, _ = environment.reset()
    for action in (2, 8, 10):
        result = environment.step(action)
        records.append(Transition(state, action, result.reward, result.state,
                                  result.action_mask, result.terminated))
        state = result.state
    first = ReplayBuffer(capacity=2, seed=7)
    second = ReplayBuffer(capacity=2, seed=7)
    for record in records:
        first.push(record)
        second.push(record)

    assert len(first) == 2
    assert first.sample(2) == second.sample(2)
    assert set(first.sample(2)) == set(records[1:])


def test_replay_rejects_non_transition_and_invalid_sample_sizes(
    observed_transition: Transition,
) -> None:
    """Protect replay type and sample-without-replacement bounds.

    Inputs:
        observed_transition: One valid record for a one-item buffer.
    Returns:
        None. The test passes when invalid pushes and samples fail.
    """

    replay = ReplayBuffer(capacity=2)
    with pytest.raises(TypeError, match="Transition"):
        replay.push(object())  # type: ignore[arg-type]

    replay.push(observed_transition)
    for size in (0, 2):
        with pytest.raises(ValueError, match="batch_size"):
            replay.sample(size)
    for size in (1.5, True):
        with pytest.raises(TypeError, match="batch_size"):
            replay.sample(size)  # type: ignore[arg-type]



@pytest.mark.parametrize("capacity,batch_size", [(2, 2), (100, 4), (100, 64)])
def test_ring_matches_original_deque_sampling(capacity: int, batch_size: int) -> None:
    """Compare logical ring sampling against the original deque algorithm.

    Inputs:
        capacity: Ring capacity including both random.sample algorithm regimes.
        batch_size: Number sampled per draw.
    Returns:
        None. Samples and RNG state match before and after repeated wraps.
    """

    import random
    from collections import deque

    replay = ReplayBuffer(capacity=capacity, seed=13)
    reference = deque(maxlen=capacity)
    random_state = random.Random(13)
    # Repeated complete flowsheets exercise eviction and both sampling paths.
    for _ in range(capacity * 2):
        environment = DistillationSequenceEnvironment()
        state, _ = environment.reset()
        for action in (2, 8, 10):
            result = environment.step(action)
            transition = Transition(state, action, result.reward, result.state,
                                    result.action_mask, result.terminated)
            replay.push(transition)
            reference.append(transition)
            state = result.state
            if len(replay) >= batch_size:
                expected = random_state.sample(list(reference), batch_size)
                assert all(a is b for a, b in zip(replay.sample(batch_size), expected))
                assert replay._random.getstate() == random_state.getstate()
