# Why conformal CBFs?

## What conformal prediction is

Conformal prediction (CP) is a distribution-free framework for turning
any black-box predictor into one that returns calibrated prediction
sets: given a user-chosen miscoverage level ε ∈ (0, 1), CP wraps the
predictor so its sets contain the true outcome with probability at
least 1 − ε, with no assumptions on the data distribution beyond
exchangeability. The canonical reference for the framework and its
finite-sample coverage proof is Shafer and Vovk (2008)
[`shafer2008conformal`] ("A Tutorial on Conformal Prediction," JMLR).

Angelopoulos and Bates (2023) [`angelopoulos2023conformal`]
("Conformal Prediction: A Gentle Introduction," Foundations and
Trends in Machine Learning) is the practical entry point for ML
readers — it walks through the split-CP, full-CP, and online-CP
variants and the regret-style guarantees that the online variant
enjoys when the data stream is non-exchangeable. What Huriot and
Sibai (2025) [`huriotsibai2025`] contributes on top is to specialise
the online-CP update to a CBF safety-filter setting: the calibration
variable becomes the per-pair conformal slack λ added to each CBF
constraint, and the resulting ε + o(1) long-term risk bound inherits
CP's distribution-freeness while binding it to a per-step safety
filter.

The exponential CBF-QP backbone (Wang, Ames, and Egerstedt (2017)
[`wangames2017`]) provides instantaneous safety guarantees but
requires tight a-priori bounds on partner behaviour. In heterogeneous
ad-hoc teamwork those bounds are unknown at deployment time.

The conformal overlay (Huriot and Sibai 2025) relaxes this
requirement: it maintains a running average-loss guarantee
(Theorem 3) with a self-tuning slack variable λ that widens when the
partner's actions are worse than expected and tightens when they are
better. The OSCBF (Morton and Pavone (2025) [`morton2025oscbf`])
resolves the multi-arm coupling into tractable per-arm QPs.

The key open theoretical question is whether the Huriot–Sibai
average-loss bound can be sharpened to a per-step bound. This is not
required for Phase 0 correctness but would strengthen the safety claim.
See [ADR-004](../reference/adrs.md) §Open questions.

## The three-layer architecture (plus braking fallback)

[ADR-004](../reference/adrs.md) §Decision pins a three-layer composition
plus a per-step backstop. Each layer addresses a different shortcoming
of the previous one.

### 1. Outer layer — exp CBF-QP backbone

Source: Wang, Ames, and Egerstedt (2017) [`wangames2017`] (Tier-2
note 37). Each pair of agents contributes one linear constraint to a
quadratic program

```
min   ||u - u_hat||^2
s.t.  A_ij u <= b_ij  (per pair (i, j))
      ||u_i||_inf <= bounds.action_norm  (per agent)
```

where `A_ij` and `b_ij` are derived from a relative-degree-1 barrier

```
h_ij = sqrt(2 * alpha_pair * max(|Δp| - D_s, 0)) + (Δp / |Δp|)^T Δv
```

The barrier ensures the closing speed stays under the brakeable speed
given the joint deceleration capacity. The per-pair budget is split by
αᵢ/(αᵢ+αⱼ) (Wang §IV) so heterogeneous actuator limits don't favour
one agent over the other.

This formulation is **dynamics-known**: it requires the agents'
acceleration limits and assumes the partner's behaviour fits into the
QP's feasibility region. The conformal overlay (next layer) loosens
that assumption.

### 2. Middle layer — conformal slack overlay

Source: Huriot and Sibai (2025) [`huriotsibai2025`] ICRA, the Tier-1
read (note 42; mislabelled "Singh" in the source CSV). A scalar slack
λ is added per-pair to the right-hand side of every CBF constraint

```
A_ij u <= b_ij + λ_ij
```

and updated each control step by Theorem 3's rule

```
λ_{k+1} = λ_k + η * (ε - l_k)
```

where `l_k` is the per-pair loss — the worst-case CBF-constraint gap
between the conformal and ground-truth constraints (Huriot and Sibai
§IV.A). The default `ε = -0.05` ([ADR-004](../reference/adrs.md)
§Decision) biases λ toward tighter constraints during contact-rich
manipulation.

The Theorem 3 bound

```
1/K' Σ l_k <= ε + (λ_1 - λ_safe + η) / (η * K')
```

is **distribution-free**: it holds even under adversarial partner
predictions. The trade-off is that it is *average-loss*, not per-step
— a single bad step can still cause a constraint violation. The
braking fallback below is the per-step backstop.

On partner identity change (`obs["meta"]["partner_id"]` mismatch with
the previous step), the conformal layer's stationarity assumption
breaks. The reset protocol restores λ to a worst-case-bound `λ_safe`
and runs a high-caution warmup window (`ε_warmup = 1.5 * ε`,
default 50 steps) so the swap transient is mitigated rather than
masked. See [ADR-004](../reference/adrs.md) risk-mitigation #2 +
[ADR-006](../reference/adrs.md) risk #3.

### 3. Inner layer — OSCBF two-level QP

Source: Morton and Pavone (2025) [`morton2025oscbf`] IROS (Tier-2
note 44). For manipulation-grade within-arm safety, the OSCBF QP
minimises

```
||W_j (q_dot - q_dot_nom)||^2 + ||W_o (J q_dot - ν_nom)||^2 + ρ ||s||^2
s.t.  A q_dot - I s <= b
      -s <= 0
```

where the joint-space and operational-space objectives both
participate, slack `s` absorbs infeasibility (Morton-Pavone §IV.D),
and the constraints encode collision-avoidance (sphere-pair
decomposition, Morton-Pavone §IV.B), joint-velocity limits, and any
custom within-arm CBFs.

Lindemann et al. (2024) [`lindemann2024safety`] (the safety survey
catalogued as Tier-2 note 45 and also referenced as "Garg/Lindemann
2024" in ADR-004) flag CBF-QP infeasibility under actuator limits as
the central open challenge in multi-robot safe control;
Morton-Pavone's slack-relaxation pattern is exactly the escape hatch
that keeps the QP feasible without abandoning the safety claim —
slack drives a quadratic penalty into the cost so the QP prefers
feasible solutions but degrades gracefully when they don't exist.

The 1 kHz solve-time target (Morton-Pavone §V; ADR-004 validation
criterion 3) lets the OSCBF run in the inner control loop on a
7-DOF Franka without falling behind the actuator-command rate.

### Per-step backstop — hard braking fallback

Source: Wang, Ames, and Egerstedt (2017) [`wangames2017`] eq. 17
hybrid braking controller. Theorem 3 is *average-loss*, not per-step;
in contact-rich
manipulation a single step of constraint violation can damage
hardware. The braking fallback bypasses the conformal QP entirely and
applies max-magnitude push-apart acceleration when the per-pair
time-to-collision drops below `τ_brake` (default 100 ms,
configurable per task).

This independence from QP feasibility is the whole point: even if
the QP is infeasible or returns a bad solution, the braking fallback
is computed from kinematic state alone and provides a guaranteed
hardware-protective response. A structural test in
`tests/property/test_braking.py` enforces that `braking.py` does not
import any QP-solver symbol — the bypass cannot be silently undone
by future refactoring.

## Worked example — outer CBF + conformal update

The minimal end-to-end loop wires the outer CBF, conformal update,
and braking fallback together. The example below is mirrored verbatim
in `tests/unit/test_docs_examples.py::test_why_conformal_walkthrough_example`
so the doc and the test cannot drift:

```python
import numpy as np

from concerto.safety.api import Bounds, SafetyState
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
cbf = ExpCBFQP(cbf_gamma=2.0)

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
if fired and override is not None:
    safe = override
else:
    # 2. Outer exp CBF-QP (Wang, Ames, and Egerstedt 2017 — wangames2017).
    safe, info = cbf.filter(
        proposed_action=nominal,
        obs={"agent_states": agents, "meta": {"partner_id": "demo"}},
        state=state,
        bounds=bounds,
    )
    # `info["constraint_violation"]` is the per-step CBF gap; it goes
    # into Table 2 of the ADR-014 three-table report but does NOT
    # drive the conformal update.
    per_step_violation = info["constraint_violation"]
    # 3. Conformal lambda update (Huriot and Sibai 2025 §IV — huriotsibai2025).
    # Driven by the prediction-gap loss against the constant-velocity
    # predictor (Theorem 3's risk bound is stated on this signal, NOT
    # on the per-step CBF gap above).
    prediction_gap = update_lambda_from_predictor(
        state,
        snaps_now=agents,
        snaps_prev=agents_prev,
        alpha_pair=2.0 * bounds.action_norm,
        gamma=2.0,
        dt=dt,
        in_warmup=False,
    )
```

After this single step the agents either receive a push-apart
brake action (if their time-to-collision dropped below 100 ms) or a
QP-projected acceleration that respects the safety barrier. The
conformal `state.lambda_` is updated in place ready for the next step.

## Communication degradation and the conformal slack

Conformal CBFs only stay tight if the safety filter receives partner
poses *fresh*. When the comm channel degrades — added latency, jitter,
or dropped packets — the filter must reason about *what it does not
know yet*. CHAMBER models this explicitly via the
[`CommDegradationWrapper`](../reference/api.md#chambercomm) that wraps
the mandatory fixed-format channel.

The wrapper is anchored to **URLLC + 3GPP Release 17** industrial-trial
data (see [ADR-006](../reference/adrs.md)) and ships six named
`DegradationProfile` rows under `chamber.comm.URLLC_3GPP_R17`:

| Profile | Latency mean (ms) | Latency std (ms) | Drop rate |
|---------|-------------------|------------------|-----------|
| `ideal` | 0.0 | 0.0 | 0.0 |
| `urllc` | 1.0 | 0.0 | 1e-6 |
| `factory` | 5.0 | 0.1 | 1e-4 |
| `wifi` | 10.0 | 1.0 | 1e-2 |
| `lossy` | 30.0 | 5.0 | 1e-2 |
| `saturation` | 100.0 | 10.0 | 1e-1 |

On every encode, the wrapper:

1. Bernoulli-drops the fresh packet with probability `drop_rate`. On
   drop, the previously visible packet is returned with each per-uid
   AoI bumped by one tick (the receiver knows it is staler now).
2. Otherwise queues the packet with delay drawn from
   `Normal(latency_mean_ms, latency_std_ms)`, clipped to
   `[0, 2 * latency_mean_ms]`. Packets due at the current tick are
   released; the most-recently-sent winner becomes the visible packet,
   with its per-uid AoI bumped by the in-queue dwell time.

The packet itself carries a per-uid **Age of Information** field
(see Ballotta and Talak (2024) [`ballotta2024aoi`] for the formal
definition). AoI is the proxy the conformal slack `λ` reads: when the
channel degrades, AoI grows, and the conformal layer widens `λ` to
absorb the additional prediction-error variance.

Two safety properties protect this composition:

- **QP saturation guard.** The wrapper never silently allows a
  configuration that would saturate the inner CBF QP solver beyond the
  ADR-004 OSCBF target (1 ms). A `ChamberCommQPSaturationWarning`
  fires under the `saturation` profile (drop ≥ 10 % or
  latency ≥ 100 ms), per [ADR-006](../reference/adrs.md) §Risks R5.
- **Determinism.** Every random draw — drop coin, latency sample —
  is derived from `concerto.training.seeding.derive_substream` so two
  resets with the same root seed produce byte-identical degradation
  traces. This keeps the Stage-2 CM spike's pre-registration auditable.

The full M2 wrapper composition that the Stage-0 smoke and Stage-2 CM
spike both use is:

```python
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
```

The fixed-format channel itself ships **no degradation** — encoding
logic stays pure and trivially testable, and degradation is composed
externally (see [ADR-003](../reference/adrs.md) §Decision).

## Reading list

The three-layer architecture and its empirical anchors are laid out
across:

- **Tier-1**: Huriot and Sibai 2025 [`huriotsibai2025`] ICRA — the
  conformal slack and Theorem 3 bound (note 42).
- **Tier-2**: Wang, Ames, and Egerstedt 2017 [`wangames2017`] T-RO —
  exp CBF-QP backbone and braking fallback (note 37); Morton and
  Pavone 2025 [`morton2025oscbf`] IROS — OSCBF two-level QP
  (note 44); Lindemann et al. 2024 [`lindemann2024safety`] — the
  multi-agent safety survey framing the open intersection (note 45,
  also referenced as "Garg/Lindemann 2024"); Ballotta and Talak 2024
  [`ballotta2024aoi`] — AoI predictor pattern (note 41).
- **ADRs**: [ADR-004](../reference/adrs.md) (filter formulation),
  [ADR-006](../reference/adrs.md) (partner-policy assumption set),
  [ADR-014](../reference/adrs.md) (three-table reporting).

For the full bibliography — `ames2017cbfqp`, `ames2019cbfsurvey`,
`shafer2008conformal`, `angelopoulos2023conformal`, `wangames2017`,
`huriotsibai2025`, and the surrounding canon — see
[the literature reference page](../reference/literature.md#3-conformal-prediction-and-conformal-control).

> Full bibliographic records for every entry on this page live in
> [`docs/reference/refs.bib`](../reference/refs.bib); the thematic
> literature map is at
> [`docs/reference/literature.md`](../reference/literature.md).
