# Decision-queue entry (for the planning kit's 00_index/DECISION_QUEUE.md)

**Transfer note.** The planning kit is not mounted on this host; per the established
operator-side pattern (REMEDIATION_LOG §7: local docs rsynced to the kit) this entry is staged
here verbatim for appending to `00_index/DECISION_QUEUE.md`. Founder-directed, 2026-06-10.

---

## Entry: recalibrate or remove the `action_linf_component` action box for filtered-training cells

- **Queued.** 2026-06-10. **Class.** ADR-004-adjacent (safety-stack bounds calibration).
  **Decision taken in the regime-alignment slice: NONE** — options enumerated only.
- **Surfaced by.** `spikes/results/stage1-failure-investigation/2026-06-09-grasp-remediation/SAFETY_INTERFERENCE_PROBE_2026-06-10.md`:
  the CBF-QP stacks per-component box rows `|u_i| ≤ Bounds.action_linf_component` on every
  solve (`concerto/safety/cbf_qp.py:870-872`), and the Stage-1b env pins
  `action_linf_component = 0.1` (`chamber/envs/stage1_pickplace.py` `build_bounds()`, an
  uncalibrated Phase-1 default) against a [−1, 1] normalised `pd_joint_delta_pos` action
  space. Consequences measured on the fix-only triplet: ~80 % of sampled action components
  saturated at every filtered step; PPO buffered unexecuted nominal actions; eval has always
  run unfiltered (train/eval dynamics mismatch).
- **Options (enumerate, do not decide here):**
  1. **Recalibrate** `action_linf_component` to the env's true per-component envelope (1.0
     for the normalised action space) — the box becomes a no-op duplicate of the env clip;
     the CBF pair rows become the only active constraint. Cheapest; needs an ADR-004
     §Revision entry naming the new value and the probe as evidence.
  2. **Remove** the box rows from the QP entirely and rely on the env's own action clipping —
     changes the QP's feasible set shape; needs ADR-004 review (the box currently guarantees
     boundedness of the QP solution).
  3. **Derive per-uid** bounds from `env.action_space` at filter construction (the same
     source `TrainedPolicyFactory` uses for partner action_dim) — most principled; largest
     surface.
  4. **Keep as-is** and document the clamp as an intentional conservative envelope — requires
     re-deriving every training-time-filtered result's interpretation and making eval apply
     the same clamp (train/eval consistency); flagged as the least defensible option by the
     probe's evidence.
- **Gating.** Any future *filtered-training* cell (including a batched-filter implementation
  at `num_envs > 1`) blocks on this decision. The A3 arm of the 2026-06-10 regime-alignment
  pre-statement supplies the isolation evidence (filter-off vs filtered, single variable).
- **Cross-refs.** ADR-004 §Decision / §Revision history 2026-05-17 (the L∞ field split);
  ADR-007 §Stage 1b Rev 17 draft §3; `PRESTATEMENT.md` §3 (A3).
