# SPDX-License-Identifier: Apache-2.0
"""Property tests for ``FixedFormatCommChannel`` round-trip + determinism (T2.3).

Covers ADR-003 §Decision (schema-typed round-trip across arbitrary state inputs)
and P6 reproducibility (same seed -> identical encode trace).

Hypothesis runs with ``derandomize=True`` so CI is byte-identical.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from chamber.comm import FixedFormatCommChannel, Pose

_uids = st.sampled_from(["a", "b", "c"])
_floats = st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)
_pose = st.builds(
    lambda x, y, z: Pose(xyz=(x, y, z), quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
    _floats,
    _floats,
    _floats,
)
_state = st.dictionaries(_uids, _pose, max_size=3).map(lambda p: {"pose": p})


@settings(derandomize=True, max_examples=200)
@given(state=_state)
def test_round_trip_preserves_pose(state: dict[str, dict[str, Pose]]) -> None:
    """ADR-003 §Decision: decode(encode(state)) returns the same pose dict."""
    channel = FixedFormatCommChannel()
    out = channel.decode(channel.encode(state))
    assert out["pose"] == state["pose"]


@settings(derandomize=True, max_examples=100)
@given(states=st.lists(_state, min_size=1, max_size=5))
def test_aoi_grows_for_omitted_uids(states: list[dict[str, dict[str, Pose]]]) -> None:
    """ADR-003 §Decision: a uid omitted from the next state's pose ages by one tick.

    Reference: AoI semantics from notes/tier2/41 (Ballotta & Talak).
    """
    channel = FixedFormatCommChannel()
    seen: set[str] = set()
    for state in states:
        prior_seen = set(seen)
        prior_aoi = channel.encode({"pose": {}})["aoi"] if seen else {}
        # restore prior aoi semantics: just track via a fresh encode run instead.
        del prior_aoi
        del prior_seen
        seen.update(state["pose"].keys())
        packet = channel.encode(state)
        for uid in state["pose"]:
            assert packet["aoi"][uid] == 0.0  # marked fresh
        for uid in seen - set(state["pose"].keys()):
            # Aged at least one tick this step (could be more from prior steps).
            assert packet["aoi"][uid] >= 1.0


@settings(derandomize=True, max_examples=50)
@given(
    states=st.lists(_state, min_size=1, max_size=5),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
def test_same_seed_reproduces_aoi_trace(
    states: list[dict[str, dict[str, Pose]]], seed: int
) -> None:
    """P6: two channels reset with the same seed produce identical AoI traces."""
    a = FixedFormatCommChannel()
    b = FixedFormatCommChannel()
    a.reset(seed=seed)
    b.reset(seed=seed)
    for state in states:
        pa = a.encode(state)
        pb = b.encode(state)
        assert pa["aoi"] == pb["aoi"]
        assert pa["pose"] == pb["pose"]
