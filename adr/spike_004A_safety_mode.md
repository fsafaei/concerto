# Spike 004A — Heterogeneous-action CBF-QP and explicit safety modes

**Type.** Pre-ADR spike note (not an ADR; not implementation-binding).
**Tracks.** ADR-004 §Open questions (per-step bound under heterogeneous
action spaces); ADR-007 Stage-1 AS spike scope (7-DOF arm vs 2-DOF
differential-drive base); ADR-INDEX footnote a.
**Author.** Farhad Safaei
**Created.** 2026-05-13
**Status.** Draft — awaiting review before promotion to a §Mode
amendment of ADR-004.

## §Problem statement

The current `concerto.safety.cbf_qp.ExpCBFQP` flattens every uid's
proposed action into one symmetric optimisation vector, derives
`n_u_per_agent` from the *first* agent's action shape, and uses a
single global `alpha_pair = 2.0 * bounds.action_norm`. Two reviewer
findings flag this as load-bearing for the architectural claims the
project makes externally.

> **Reviewer P0-2 (heterogeneous action dimension is mandatory for
> the AS axis).** "The current `ExpCBFQP` implementation treats the
> action vectors of all agents as a single homogeneous flat
> optimisation variable and returns safe actions for every uid. This
> conflicts with CHAMBER's headline claim that the AS axis (7-DOF
> arm vs 2-DOF mobile base) requires per-agent action dimension."

> **Reviewer P0-3 (black-box partner setting requires an ego-only
> filter at deployment).** "The black-box / ad-hoc partner setting
> requires the deployment-time filter to optimise only the ego
> agent's action; the partner's motion must enter as a predicted
> disturbance. A filter that decides both agents' actions assumes
> the partner is co-controllable, which is precisely the property
> the ad-hoc-teamwork brief rules out."

Both findings are transcribed from the spike brief that opened this
PR; the source review log lives in the founder's local planning kit
and is not mirrored into the repo. They point at the same root
cause: the QP variable layout silently encodes assumptions
(homogeneity; centralised oracle control) that neither the AS axis
of ADR-007 nor the partner-policy contract of ADR-006 actually
admit. The implication is a refactor in two coupled directions —
(i) the QP variable must carry per-agent action dimension, and (ii)
the QP must accept an explicit "what is co-optimisable here?" flag
so callsites cannot accidentally request the oracle solve at
deployment.

ADR-004 §Risks already foreshadows the per-agent dimensionality
issue at the *fallback* boundary: the `JacobianEmergencyController`
placeholder raises `NotImplementedError` so that 7-DOF uids loud-fail
rather than receive a Cartesian-shaped vector. The QP backbone needs
the analogous treatment.

## §Three-mode taxonomy

We introduce a `SafetyMode` enum carried on the filter (constructor
argument; not a per-call flag — flipping mode mid-episode is an
anti-pattern that the type system should help catch). Each mode has
an explicit contract for *who is being optimised over* and *what
role the partner plays*.

| Mode | QP decision variable | Partner role | Feasibility | When allowed |
|---|---|---|---|---|
| `EGO_ONLY` | ego action only (length `ego.action_dim`) | Predicted disturbance on the constraint RHS — partner's predicted Cartesian acceleration is treated as a known drift term, not a variable | Always solvable when the predicted disturbance does not push the constraint into infeasibility; if it does, `ConcertoSafetyInfeasible` raises and the braking fallback fires per ADR-004 risk-mitigation #1 | **Deployment** (ad-hoc / black-box partner; ADR-006 §Decision). The default. |
| `CENTRALIZED` | concatenated per-agent actions, with per-uid slot widths from `control_models[uid].action_dim` (no "first uid's shape" fallback) | Co-controllable agent | Oracle co-control envelope (a valid solve exists iff *some* joint action keeps all pairs above the barrier) | **Oracle / ablation only.** Used as the upper-bound baseline in ADR-014's three-table report (Table 2 oracle row). Never used at evaluation against ad-hoc partners. |
| `SHARED_CONTROL` | concatenated per-agent actions, but the partner's adjustment magnitude is restricted by an additional `partner_action_bound: float` parameter | Partially co-controllable agent: the ego decides freely, the partner's action is constrained to a ball around its *proposed* action | Same QP as `CENTRALIZED` plus a per-pair L-infinity box on the partner's slot | **Lab-only baseline** for the reviewer's P0-3 mode 3 — measures how much of the `CENTRALIZED` headroom comes from co-control budget vs from partner co-operation. Never appears in deployment configs. |

### Why an enum rather than separate classes

`EGO_ONLY` is the production interface; `CENTRALIZED` is the ablation
baseline; `SHARED_CONTROL` is a measurement instrument. They share
the same constraint geometry (the same pairwise CBF rows over
Cartesian acceleration), the same conformal `lambda` state, and the
same telemetry payload. Splitting them into three classes would
duplicate the row construction without splitting any of the parts
that actually differ; an enum keeps the shared code in one place and
makes the mode visible at the constructor callsite.

### Why mode is constructor-scoped, not per-call

`SafetyState.lambda_` is sized per pair, and the pairing is
implicit in the uid ordering. `EGO_ONLY` builds one pairwise row per
`(ego, partner)` pair; `CENTRALIZED` builds the full pairwise table.
Flipping mode mid-episode would silently invalidate the conformal
state. Making mode a constructor argument lets the partner-swap
reset path (ADR-004 risk-mitigation #2) stay structurally the same;
a mode change requires constructing a new filter and a fresh
`SafetyState`, which is what we actually want to express.

## §Per-agent control model

The current `_pair_constraint_row` builds the row over the joint
action vector directly. For a double-integrator agent whose action
is Cartesian acceleration, this is correct because the action-space
basis *is* the Cartesian basis. For a 7-DOF arm whose action is
joint torque (or joint velocity), the same row construction would
emit Cartesian-shaped coefficients into a 7-D joint slot — exactly
the failure mode ADR-004 §Risks flags at the fallback boundary.

We resolve this by introducing an `AgentControlModel` Protocol that
mediates between each agent's *action space* and the shared
*safety space* (Cartesian acceleration of the agent's safety body).

```python
class AgentControlModel(Protocol):
    uid: str
    action_dim: int
    position_dim: int  # Cartesian dim of the agent's safety body

    def action_to_cartesian_accel(
        self, state: AgentSnapshot, action: FloatArray
    ) -> FloatArray: ...

    def cartesian_accel_to_action(
        self, state: AgentSnapshot, cartesian_accel: FloatArray
    ) -> FloatArray: ...
```

The CBF row construction moves to Cartesian / safety space. The
linear constraint `n_hat^T (ddot p_i - ddot p_j) >= ...` is built in
Cartesian acceleration variables, then projected through each
agent's control map to obtain its action-space coefficients. For
slot `i`:

```
row_action_i = -n_hat^T @ J_i(state_i)
```

where `J_i = d(cartesian_accel_i) / d(action_i)` is the Jacobian of
the agent-specific action→Cartesian map evaluated at the current
state. The `cartesian_accel_to_action` right-inverse exists so the
QP projection back into action space is well-defined when the action
space is over-actuated (Jacobian has more columns than rows); for an
exactly-actuated double-integrator the right-inverse is the
identity.

### Reduction to the current double-integrator case

For a double-integrator agent with `action_dim == position_dim` and
the trivial map `action_to_cartesian_accel(state, u) = u`:

- `J_i = I` (identity of shape `(position_dim, position_dim)`).
- `row_action_i = -n_hat^T` — element-wise identical to the row
  emitted by the current `_pair_constraint_row` in `cbf_qp.py:189`.
- `cartesian_accel_to_action` is the identity, so the QP variable
  layout coincides with today's layout when every agent uses
  `DoubleIntegratorControlModel(uid, dim)`.

This is the migration shim: every existing test that constructs a
homogeneous 2-D pair (Wang-Ames-Egerstedt §V toy crossing,
`test_well_separated_agents_pass_through_proposed_action`,
`test_head_on_collision_avoidance_wang_ames_egerstedt_crossing`,
`test_lambda_relaxes_constraint_brings_safe_action_closer_to_proposal`,
the docs example in `docs/explanation/why-conformal.md`) passes a
`DoubleIntegratorControlModel` per uid and runs unchanged through
the new QP path in `CENTRALIZED` mode. The geometry-level invariants
exercised by `tests/unit/test_geometry.py` are untouched — the
geometry module (sphere decomposition, signed pair distance,
gradient) operates entirely in Cartesian space and does not depend
on the action representation.

### Per-agent alpha via budget_split

The symmetric `alpha_pair = 2.0 * bounds.action_norm` is replaced by
a per-pair sum of the two agents' Cartesian acceleration capacities:

```
alpha_i_cart = control_models[uid_i].max_cartesian_accel(bounds)
alpha_j_cart = control_models[uid_j].max_cartesian_accel(bounds)
alpha_pair = alpha_i_cart + alpha_j_cart
```

The proportional Wang-Ames-Egerstedt 2017 §IV split
(`concerto.safety.budget_split.ProportionalBudgetSplit`) is already
implemented; ADR-004 §Rationale already names this split as the
mechanism for "heterogeneous embodiments in §6.2". The QP itself
still operates centrally over the joint variable in
`CENTRALIZED` / `SHARED_CONTROL` modes — the per-agent fraction is
used to bound each agent's slot in the action-space L-infinity box
and to set the per-pair alpha used to build `h_ij`. In `EGO_ONLY`
mode the partner's slot does not exist as a variable, so the split
is consumed only to size the constraint RHS contribution from the
partner's predicted disturbance.

The `RelativeDegreeAwareBudgetSplit` stub (already in the codebase)
remains a Phase-1 deliverable; this spike does not resolve OQ #3
(mixed-relative-degree pairs).

### What does *not* change

- `concerto.safety.geometry` — sphere decomposition and signed-pair-
  distance primitives are kinematics-agnostic.
- `concerto.safety.oscbf` — the inner-loop joint-space filter already
  takes a Jacobian and operates over joint variables. OSCBF does not
  consume `AgentControlModel`; it is the *per-arm* filter, whereas
  `AgentControlModel` lives at the *inter-agent* boundary. The spec
  call to "pass `test_oscbf.py` through `DoubleIntegratorControlModel`"
  is a no-op for that reason, and the memo records this so the
  migration plan stays honest.
- `concerto.safety.conformal` — operates on `pair_h_value` in
  Cartesian space; unchanged.
- `concerto.safety.braking` and `concerto.safety.emergency` — already
  embodiment-aware via the `EmergencyController` Protocol.

## §Migration plan

### Files that change

| File | Change | Risk |
|---|---|---|
| `src/concerto/safety/api.py` | Add `SafetyMode` enum, `AgentControlModel` Protocol, `DoubleIntegratorControlModel`, `JacobianControlModel` skeleton (Stage-1 deliverable; `NotImplementedError` until a Jacobian is supplied) | Public surface change; downstream importers gain new names but no name is removed |
| `src/concerto/safety/cbf_qp.py` | `ExpCBFQP.__init__` gains `mode` (default `EGO_ONLY`) and required `control_models: dict[str, AgentControlModel]`. `ExpCBFQP.filter` signature splits per mode: `EGO_ONLY` takes `(proposed_action: FloatArray, ego_uid: str, partner_predicted_states: dict[str, AgentSnapshot])` and returns `(safe_ego_action, info)`; `CENTRALIZED` and `SHARED_CONTROL` keep the dict-of-actions shape | Breaking change to every callsite of `ExpCBFQP.filter`; explicit mode at every callsite is the design requirement |
| `src/concerto/safety/__init__.py` | Re-export the new names | Low |
| `src/chamber/benchmarks/training_runner.py` (and env adapters) | When a downstream config wires a safety filter, the env adapter must construct the `control_models` dict from its per-robot metadata. Phase-0 training_runner does not currently instantiate a filter, so the runner-level change is small — the adapter-level change is where the per-robot metadata gets read | Low — the wiring is local to each env adapter |

### Tests that change

| Test | Change |
|---|---|
| `tests/property/test_cbf_qp.py` | Every `ExpCBFQP()` call gains explicit `mode=SafetyMode.CENTRALIZED` and `control_models={uid: DoubleIntegratorControlModel(uid, dim=2)}`. Invariants hold unchanged through the migration shim |
| `tests/integration/test_safety_in_loop.py` | Same migration shim. The three-table report's oracle row stays in `CENTRALIZED`; the deployment row gains a sibling `EGO_ONLY` smoke variant per the spec's validation note |
| `tests/integration/test_partner_swap.py::test_partner_swap_consumer_gates_on_partner_id_none` | Switch to explicit `CENTRALIZED` mode; the partner-id-None contract is unchanged |
| `tests/unit/test_docs_examples.py::test_why_conformal_walkthrough_example` and `docs/explanation/why-conformal.md` | Doc code block updated in lock-step so the test mirrors the doc. The published example moves to `EGO_ONLY` (the deployment story is what the doc explains); a sidebar block shows the `CENTRALIZED` oracle |
| `tests/unit/test_control_model.py` (new) | Identity invariant for `DoubleIntegratorControlModel`; `NotImplementedError` for the `JacobianControlModel` skeleton until a Jacobian is supplied |
| `tests/integration/test_ego_only_filter.py` (new) | Two agents of distinct `action_dim` (4-D ego, 2-D partner). Asserts QP variable size = ego `action_dim`; partner's predicted motion enters via the RHS; returned safe action shape = ego `action_dim`, not concatenated |
| `tests/property/test_centralized_heterogeneous.py` (new) | Three agents of distinct `action_dim` (2, 4, 7). Asserts the QP builds, solves, returns shape-correct per-agent actions |

### Tests that are deliberately left unchanged

- `tests/unit/test_geometry.py` — operates in Cartesian space only.
- `tests/property/test_oscbf.py` — OSCBF is per-arm joint-space and
  already Jacobian-parameterised; it does not import `ExpCBFQP` and
  does not need `AgentControlModel`. The spec's "pass through
  `DoubleIntegratorControlModel`" instruction does not apply here;
  the surrounding invariants ("OSCBF respects joint-space limits";
  "slack relaxation under conflict") hold unchanged.
- `tests/property/test_braking.py`, `tests/property/test_braking_multipair.py` —
  embodiment-aware already via `EmergencyController`; no change.
- `tests/property/test_conformal_loss.py`, `tests/unit/test_conformal.py` —
  operate on Cartesian-space `pair_h_value` and do not depend on
  the QP variable layout.

### Tests that need a temporary xfail

None at the spike scope. Stage-1 AS spike will exercise the
`JacobianControlModel` against a real 7-DOF arm; until then the
class raises `NotImplementedError` on `action_to_cartesian_accel`
unless a Jacobian callable is supplied, which is the same
loud-fail discipline ADR-004 §Risks established for
`JacobianEmergencyController`. No silent fallback path.

### Configs that change

None at Phase-0. `concerto.safety.budget_split.make_budget_split`
still defaults to `"proportional"`; no new config knobs.

## §Open questions

Inherited from ADR-004 (this spike does **not** resolve any of them):

1. **Per-step safety bound under heterogeneous action spaces**
   (ADR-004 Open Question #1; ADR-INDEX footnote a). The conformal
   layer is an average-loss bound (Huriot & Sibai 2025 Theorem 3).
   Sharpening to per-step under heterogeneous embodiments is gated
   by the Stage-1 AS spike. This spike is the *necessary
   refactor*; the *sufficient bound* still requires the spike-and-
   theory follow-up.

2. **Predictor warm-up for a black-box VLA partner** (ADR-004 Open
   Question #2). The `EGO_ONLY` mode consumes
   `partner_predicted_states`, but the *quality* of that prediction
   under a multi-modal VLA action distribution is unchanged by this
   refactor. The constant-velocity stub
   (`concerto.safety.conformal.constant_velocity_predict`) remains
   the Phase-0 predictor; Phase-1 will land the AoI-conditioned
   variant per ADR-003 / ADR-006.

3. **Mixed-relative-degree pairs** (ADR-004 Open Question #3;
   `RelativeDegreeAwareBudgetSplit` stub). The Jacobian-aware row
   construction is a *necessary* condition for handling
   velocity-controlled mobile bases alongside torque-controlled arms,
   but it is not *sufficient*: the relative degree of the barrier
   differs across embodiment classes, and the proportional split is
   only correct when both agents share the relative degree. The
   Phase-1 `RelativeDegreeAwareBudgetSplit` lands together with the
   §6.2 Phase-2 CBF design document.

## §Review note

Per project rule #5 ("pre-registration discipline") and the brief's
explicit instruction to "share and request review before proceeding
to code", this memo is the gate. If approved as-is, Tasks 1–5 of
the spike brief proceed. If iterated, the memo is the diff target,
not the code.
