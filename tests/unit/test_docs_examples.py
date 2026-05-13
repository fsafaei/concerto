# SPDX-License-Identifier: Apache-2.0
"""Code-block tests for the published docs (T3.12).

Plan/03 §4 T3.12: "Code-block-test the example." Each function in
this module mirrors a code block in ``docs/explanation/*.md``
verbatim (modulo the assertion lines and the surrounding test
scaffolding) so the doc and the runtime contract cannot drift —
when either changes, the test fails or a reviewer notices the
mismatch in the diff.
"""

from __future__ import annotations

import numpy as np


def test_why_conformal_walkthrough_example() -> None:
    """Mirror of ``docs/explanation/why-conformal.md`` "Worked example" code block.

    Validates that the three-layer composition example (braking
    fallback + outer exp CBF-QP + conformal lambda update) runs
    end-to-end on a head-on 2-agent scenario.
    """
    # Test-scaffolding imports (outside the doc-mirrored block).
    from typing import cast

    # --- BEGIN: code block from docs/explanation/why-conformal.md ---
    from concerto.safety.api import (
        AgentControlModel,
        Bounds,
        DoubleIntegratorControlModel,
        FloatArray,
        SafetyState,
    )
    from concerto.safety.braking import maybe_brake
    from concerto.safety.cbf_qp import AgentSnapshot, ExpCBFQP
    from concerto.safety.conformal import update_lambda_from_predictor

    dt = 0.05
    bounds = Bounds(
        action_norm=5.0,
        action_rate=0.5,
        comm_latency_ms=1.0,
        force_limit=20.0,
    )
    state = SafetyState(
        lambda_=np.zeros(1, dtype=np.float64),
        epsilon=-0.05,
        eta=0.01,
    )
    # The deployment-time filter optimises only the ego agent's action;
    # the partner enters as a predicted disturbance. The explicit
    # CENTRALIZED variant below is used here to keep the walkthrough
    # symmetric in both agents (both shown as decision variables);
    # production deployments use ExpCBFQP.ego_only(...).
    control_models: dict[str, AgentControlModel] = {}
    control_models["a"] = DoubleIntegratorControlModel(uid="a", action_dim=2)
    control_models["b"] = DoubleIntegratorControlModel(uid="b", action_dim=2)
    cbf = ExpCBFQP.centralized(
        cbf_gamma=2.0,
        control_models=control_models,
    )

    agents = {
        "a": AgentSnapshot(
            position=np.array([-1.0, 0.0], dtype=np.float64),
            velocity=np.array([1.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
        "b": AgentSnapshot(
            position=np.array([1.0, 0.0], dtype=np.float64),
            velocity=np.array([-1.0, 0.0], dtype=np.float64),
            radius=0.2,
        ),
    }
    # Snapshots from one control step ago, used to score the constant-velocity
    # predictor against the actual ``agents`` state — Huriot & Sibai 2025 §IV.A.
    agents_prev = {
        uid: AgentSnapshot(
            position=snap.position - snap.velocity * dt,
            velocity=snap.velocity.copy(),
            radius=snap.radius,
        )
        for uid, snap in agents.items()
    }
    nominal = {
        "a": np.zeros(2, dtype=np.float64),
        "b": np.zeros(2, dtype=np.float64),
    }

    # 1. Per-step braking fallback (kinematic backstop).
    override, fired = maybe_brake(nominal, agents, bounds=bounds)
    safe: dict[str, FloatArray]
    if fired and override is not None:
        safe = override
    else:
        # 2. Outer exp CBF-QP (Wang-Ames-Egerstedt 2017).
        raw_safe, info = cbf.filter(
            proposed_action=nominal,
            obs={"agent_states": agents, "meta": {"partner_id": "demo"}},
            state=state,
            bounds=bounds,
        )
        safe = cast("dict[str, FloatArray]", raw_safe)
        # `info["constraint_violation"]` is the per-step CBF gap; it goes
        # into Table 2 of the ADR-014 three-table report but does NOT
        # drive the conformal update.
        per_step_violation = info["constraint_violation"]
        # 3. Conformal lambda update (Huriot & Sibai 2025 §IV).
        prediction_gap = update_lambda_from_predictor(
            state,
            snaps_now=agents,
            snaps_prev=agents_prev,
            alpha_pair=2.0 * bounds.action_norm,
            gamma=2.0,
            dt=dt,
            in_warmup=False,
        )
        # --- END: code block from docs/explanation/why-conformal.md ---
        # The two telemetry handles must be live arrays of the right
        # shape so the doc's claim about them is checked here.
        assert per_step_violation.shape == (1,)
        assert prediction_gap.shape == (1,)

    # The example runs without exception; verify the safe action is
    # well-shaped and finite so a reader can trust the doc's claim
    # that the loop produces usable output.
    assert set(safe.keys()) == {"a", "b"}
    for uid in ("a", "b"):
        assert safe[uid].shape == (2,)
        assert np.all(np.isfinite(safe[uid]))


def test_why_conformal_comm_wrapper_example() -> None:
    """Mirror of ``docs/explanation/why-conformal.md`` comm-wrapper code block.

    Validates that the Stage-2 CM spike's composed channel constructs
    cleanly with the documented arguments.
    """
    # --- BEGIN: code block from docs/explanation/why-conformal.md ---
    from chamber.comm import (
        URLLC_3GPP_R17,
        CommDegradationWrapper,
        FixedFormatCommChannel,
    )

    channel = CommDegradationWrapper(
        FixedFormatCommChannel(),
        URLLC_3GPP_R17["factory"],
        tick_period_ms=1.0,
        root_seed=0,
    )
    # --- END: code block from docs/explanation/why-conformal.md ---

    # The published example must produce a usable channel object;
    # verify the public surface the doc implicitly promises.
    assert channel is not None
    assert hasattr(channel, "encode")
    assert hasattr(channel, "decode")
    assert hasattr(channel, "reset")


def test_add_partner_walkthrough_subclass_block() -> None:
    """Mirror of ``docs/how-to/add-partner.md`` "Subclass + register" block (T4.10).

    Validates that the recipe published in the how-to compiles, registers
    the class, and instantiates via ``load_partner`` exactly as written.
    """
    from chamber.partners import registry as registry_module

    # Use an isolated registry so the temporary class doesn't leak.
    fresh: dict[str, type] = {}
    saved = registry_module._REGISTRY
    registry_module._REGISTRY = fresh
    try:
        # --- BEGIN: code block from docs/how-to/add-partner.md ---
        from chamber.partners import PartnerBase, register_partner

        @register_partner("my_new_partner")
        class MyNewPartner(PartnerBase):
            def reset(self, *, seed: int | None = None) -> None:
                del seed

            def act(self, obs, *, deterministic=True):
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        # --- END ---

        # --- BEGIN: code block from docs/how-to/add-partner.md ---
        from chamber.partners import PartnerSpec, load_partner

        spec = PartnerSpec(
            class_name="my_new_partner",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
            extra={"uid": "fetch"},
        )
        partner = load_partner(spec)
        # --- END ---
    finally:
        registry_module._REGISTRY = saved

    assert isinstance(partner, MyNewPartner)
    assert partner.spec is spec
    assert len(spec.partner_id) == 16


def test_hello_spike_tutorial_code_block(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Mirror of ``docs/tutorials/hello-spike.md`` "Run it" code block (T4b.15).

    Plan/03 §4 T3.12 / plan/05 §4 T4b.15: the published code block is a
    contract — when the doc or the runtime API changes, this test fails.
    The tutorial's Python script runs unchanged here (modulo the
    ``repo_root`` / ``configs_dir`` substitution: the doc points at the
    user's CONCERTO checkout, the test points at this repo).
    """
    # Resolve the project's configs/ directory (siblings of src/ + tests/).
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "configs"

    # --- BEGIN: code block from docs/tutorials/hello-spike.md ---
    from chamber.benchmarks.training_runner import run_training
    from concerto.training.config import load_config

    cfg = load_config(
        config_path=configs_dir / "training" / "ego_aht_happo" / "mpe_cooperative_push.yaml",
        overrides=[
            "total_frames=1000",
            "checkpoint_every=500",
            "happo.rollout_length=250",
            f"artifacts_root={tmp_path / 'hello_spike_artifacts'}",
            f"log_dir={tmp_path / 'hello_spike_logs'}",
        ],
    )
    curve = run_training(cfg, repo_root=repo_root)
    # --- END ---

    # The published example must produce a usable RewardCurve. Verify the
    # public-surface promises the tutorial makes: 1000 steps; episodes count
    # > 0; 2 checkpoints (at frames 500 and 1000); each checkpoint paired
    # with its sidecar.
    assert len(curve.per_step_ego_rewards) == 1000
    assert len(curve.per_episode_ego_rewards) > 0
    assert len(curve.checkpoint_paths) == 2
    for path in curve.checkpoint_paths:
        assert path.exists()
        assert (path.parent / (path.name + ".json")).exists()
    assert len(curve.run_id) == 16
