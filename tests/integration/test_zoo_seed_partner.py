# SPDX-License-Identifier: Apache-2.0
"""FrozenHARLPartner end-to-end on the published zoo-seed artefact (T4b.14; B4).

Companion to :mod:`tests.reproduction.test_zoo_seed_artifact` — that
test verifies the SHA-256 manifest; this one drives the loader through
:class:`chamber.partners.frozen_harl.FrozenHARLPartner` so the
``act(obs)`` contract is exercised against the real GPU-produced
weights (plan/05 §6 #5 / plan/04 §3.6).

What this test pins (extends the M4 gate from plan/04 §6 #2):

1. The published ``.pt`` payload + sidecar round-trip through
   :func:`concerto.training.checkpoints.load_checkpoint`.
2. The HARL :class:`StochasticPolicy` rebuild via shape-inference
   succeeds on the real weights' layer keys + shapes (no silent
   layout drift between :class:`chamber.benchmarks.ego_ppo_trainer.EgoPPOTrainer`'s
   ``state_dict`` and :class:`FrozenHARLPartner`'s loader).
3. ``act(obs)`` returns a finite vector of the actor's inferred
   action_dim under a flat ``state`` obs of the actor's inferred
   obs_dim.
4. Every parameter has ``requires_grad is False`` post-load.

Skip-when-absent: the canonical artefact is not committed to the
public repo. The Tier-1 integration test in
:mod:`tests.integration.test_draft_zoo` exercises the same loader
against fixture-staged weights and runs on every PR — this test is
the GPU-host fast-lane confirmation.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from chamber.partners.frozen_harl import FrozenHARLPartner
from chamber.partners.selection import make_phase0_draft_zoo

if TYPE_CHECKING:
    from chamber.partners.api import PartnerSpec

pytestmark = pytest.mark.slow

_ARTIFACTS_ROOT_ENV: str = "CONCERTO_ARTIFACTS_ROOT"
_DEFAULT_ARTIFACTS_ROOT: Path = Path("./artifacts")


def _harl_spec() -> PartnerSpec:
    """Return the canonical frozen-HARL spec from the Phase-0 draft zoo."""
    zoo = make_phase0_draft_zoo()
    [spec] = [s for s in zoo if s.class_name == "frozen_harl"]
    return spec


def _resolve_artifacts_root() -> Path:
    raw = os.environ.get(_ARTIFACTS_ROOT_ENV)
    return Path(raw) if raw else _DEFAULT_ARTIFACTS_ROOT


def _payload_path(spec: PartnerSpec) -> Path:
    assert spec.weights_uri is not None
    relative = spec.weights_uri.removeprefix("local://")
    return _resolve_artifacts_root() / relative


def test_frozen_harl_partner_acts_on_published_zoo_seed() -> None:
    """plan/05 §6 #5: load the real ``.pt`` and run ``act(obs)`` on a zero state.

    The obs_dim is inferred from the loaded actor's
    ``base.feature_norm.weight``; the test builds a matching zero
    ``obs["agent"][uid]["state"]`` array and asserts the action is
    finite and of the actor's inferred action_dim.
    """
    spec = _harl_spec()
    payload = _payload_path(spec)
    if not payload.exists():
        pytest.skip(
            f"Zoo-seed artefact not present at {payload}. Run `make zoo-seed-pull` to "
            f"fetch it (or set CONCERTO_ARTIFACTS_ROOT to a directory that contains "
            f"the staged artefact)."
        )

    partner = FrozenHARLPartner(spec)
    partner.reset(seed=0)

    # _ensure_loaded caches obs/action dims on the wrapper; reach in to size the
    # synthetic obs for this test. Production callers use the wrapper-chain obs.
    actor = partner._ensure_loaded()
    assert actor is not None
    obs_dim = partner._obs_dim
    action_dim = partner._action_dim
    assert obs_dim > 0
    assert action_dim > 0

    for name, param in actor.named_parameters():
        assert param.requires_grad is False, f"parameter {name} still requires grad"

    state = np.zeros(obs_dim, dtype=np.float32)
    obs = {"agent": {spec.extra["uid"]: {"state": state}}}
    action = partner.act(obs)
    assert action.shape == (action_dim,)
    assert action.dtype == np.float32
    assert np.all(np.isfinite(action))
