# How-to: Add a new safety filter

A safety filter consumes a nominal action and an observation and returns
a *projected* action that lies in the safe set (ADR-004 §Decision). The
public surface for third-party filters is split by mode — pick the
Protocol that matches the filter's intended deployment shape (ADR-004
§Public API; spike_004A §Three-mode taxonomy).

## Pick a Protocol

There are two filter Protocols in `concerto.safety.api`:

- `EgoOnlySafetyFilter` — **deployment / ad-hoc / black-box-partner.**
  The QP decides the ego agent's action only; the partner's motion
  enters as a predicted disturbance on the constraint RHS. `filter()`
  takes a single `FloatArray` ego action plus `ego_uid` and
  `partner_predicted_states`, and returns a `FloatArray`. This is the
  Protocol every ADR-009 partner-zoo evaluation uses.
- `JointSafetyFilter` — **oracle / ablation baseline (`CENTRALIZED`)
  and lab-only co-control (`SHARED_CONTROL`).** The QP decides every
  uid's action; `filter()` takes a `dict[uid, FloatArray]` and returns
  a dict of the same shape. `SHARED_CONTROL` additionally requires
  `partner_action_bound` at call time.

The pre-refactor single `SafetyFilter` alias still resolves (as
`Union[EgoOnlySafetyFilter, JointSafetyFilter]`) but emits a
`DeprecationWarning` on first use. New code targets the typed
Protocols directly; the alias will be removed in 0.3.0.

## Steps

1. Implement either `EgoOnlySafetyFilter` or `JointSafetyFilter` from
   `concerto.safety.api`. Both are `@runtime_checkable`, so a duck-
   typed class that defines `reset` and `filter` with the matching
   signature is accepted.
2. If you are wrapping the built-in exp CBF-QP backbone, construct it
   via the mode-specific typed classmethod so static checkers narrow
   the return type to the right Protocol:

   ```python
   from concerto.safety.cbf_qp import ExpCBFQP

   ego_filter = ExpCBFQP.ego_only(control_models=models)
   oracle_filter = ExpCBFQP.centralized(control_models=models)
   shared_filter = ExpCBFQP.shared_control(control_models=models)
   ```

3. Register the filter in `concerto.safety.__init__` if it is shipped
   as part of the method package; third-party plugins live in their
   own packages and depend on `concerto.safety.api` only.
4. Write a property test verifying the feasibility / objective-bound
   invariant (see `plan/08-quality-and-process.md §3` and the existing
   `tests/property/test_cbf_qp.py`).

The CBF formulation is encapsulated behind the QP constraint-generation
module per ADR-004 §Reversibility; swapping in HO-CBF or MPC in Phase 3
only requires reimplementing the per-pair row builder, not the
Protocol surface.
