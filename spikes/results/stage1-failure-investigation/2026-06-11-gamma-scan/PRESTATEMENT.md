# γ-scan pre-statement — closing the C1 question

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT — awaiting founder sign-off. Frozen verbatim at launch (I8 from launch
onward); **nothing launches without explicit founder approval of this document.**
**One question.** Does a γ between the measured extremes (0.8: place 30/30, success 0/90;
0.99: place 16/30, success 2/16 conditional) give the gate a regime where BOTH place and
success consolidate — or is the place↔static trade unbridgeable by the discount alone?
**Scope guard (I1).** A/B arms are Rev 17-class regime knobs (per-task discount; budget) —
no curriculum, no reward change, no entropy schedule, no demonstrations, no predicate or
threshold edits. Option C (PBRS settle term) is a lever and is NOT in this plan; it is drafted
(never launched) only on this plan's no-γ\* branch.

## Grounding

- `../2026-06-10-success-static-probe/SUCCESS_STATIC_PROBE_2026-06-10.md` — the C1-pure
  verdict: hold-qvel median 0.52 rad/s, never < 0.2 in 7,530 placed steps; static_rate
  declining over training; P(success | placed) monotone in γ. The mechanism note for the
  record (probe §C1-confirmed): the static term `1 − tanh(5·qvel)` is near-flat between the
  observed hold speed (≈ 0.52 rad/s, term ≈ 0.01) and ≈ 0.35 rad/s — a multi-step
  near-zero-gradient valley between the learned hold and the paying region, which is why the
  effect is monotone in γ and why static_rate declines as the γ=0.8 policy optimises.
- `../2026-06-10-success-static-probe/OPTIONS_NOTE_C1_2026-06-10.md` — options A (this scan's
  A arms) and B (this scan's B arm), founder-selected for execution as a batch.
- ADR-007 §Stage 1b Rev 17 (regime knobs are implementation-details-of-the-cell; condition
  symmetry mandatory only for gate-facing runs — this scan is hetero-only investigation, the
  gate spec inherits its γ\* condition-symmetrically); ADR-002 (provenance; GPU 95 %-CI caveat).
- **Precondition: PR #217 merged** — the scan's archives must carry collision-proof run_ids
  (B2 / #214; seven same-seed-adjacent cells at one commit) and the terminal-checkpoint flush
  (W1) so each run's final policy lands on disk. The scan launches from post-merge `main`.

## Arms (draft; frozen at sign-off)

All arms: N=1024, fix-only (Rev 15+16), training-time filter off (Rev 17 operator-override
posture), AS-hetero (`stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent`),
`rollout_length=32` (32,768-transition updates), `batch_size=4096`, hidden 256, lr 3e-4 —
identical to the regime-alignment A1/A2 cells except the scanned knob.

| arm | γ | frames | seeds | role |
|---|---|---|---|---|
| **A-0.9** | 0.90 | 20M | 0/1/2 | discount mid-point (static tail ≈ 2× the γ=0.8 value) |
| **A-0.95** | 0.95 | 20M | 0/1/2 | discount mid-point (static tail ≈ 4×) |
| **B** | 0.80 | **40M** | 0 | closes the budget-not-discount branch; the declining static_rate trend predicts null |

Anchors (no new runs): regime-alignment A1 (γ=0.8, 20M ×3) and A2 (γ=0.99, 20M ×1).

Run order (chained; halt-without-retry on failure; no mid-run changes):
A-0.9 s0 → s1 → s2 → A-0.95 s0 → s1 → s2 → B.

## Instrument (per run)

- **Cold deterministic eval, 30 episodes per seed** (prereg'd Rev 16 init, forced cold,
  `num_envs=1`, unfiltered closure — identical instrument to the regime-alignment and probe
  records), with **per-step logging** of the probe's measures: arm max|qvel|, both predicate
  flags, cube–goal distance. Reported per arm: the grasp/place/success ladder
  (`ever_grasped` / `ever_placed` / `ever_success` per 30), **P(static∧placed | placed)** at
  step level, and the **within-episode hold-qvel margin distribution** (median / p10 / min vs
  the 0.2 threshold — does the hold speed move down with γ, even where success stays 0?).
- **Training trajectory:** static_rate over windows (does the γ=0.9/0.95 policy stop
  unlearning stillness?), grasp/place/success window rates, PPO health (`advantage_std`,
  `dist_entropy`, `value_mean` — A2's γ=0.99 pathology signature (value ≈ 51, entropy ≈ 5.1)
  is the watch-marker for the high-γ side).
- Artifacts per run (I8): results/signature JSON (collision-proof run_id), per-window JSONL,
  per-step cold-eval JSONL, trimmed launch log; `SHA256SUMS.txt` regenerated at the end.

## Decision rule (draft; founder may amend; FROZEN at launch)

**Gate-viable γ\*:** any scanned γ with **≥2/3 seeds at cold place ≥ 27/30 AND cold success
≥ 3/30**. If both A arms qualify, tie-break by higher pooled success, then higher pooled
place.

- **γ\* found →** the C1 question closes; `GATE_REGIME_SPEC` is drafted at γ\* (Phase 4a;
  the gate spike still launches only on separate founder approval).
- **B alone produces success ≥ 3/30** (at γ=0.8/40M) → evidence for budget-not-discount —
  **surfaced to the founder rather than auto-decided** (it would reopen the cheap branch at
  the gate's place-friendly γ; the founder weighs it against any qualifying A arm).
- **No arm qualifies →** the scan's measured margins (hold-qvel distributions per γ) become
  the design-evidence base for option C, and the option-C pre-statement is **drafted only**
  (Phase 4b; Ng–Harada–Russell PBRS settle term; own ADR note). Nothing launches.

A-arm seeds that fail the place bar but pass success (or vice versa) count as not qualifying;
partial credit is deliberately excluded — the gate needs both rungs simultaneously.

## Budget

At measured throughput (11.1k steps/s sustained at N=1024): 6 × 20M ≈ 3.3 h + 1 × 40M ≈
1.1 h ≈ **4.5 h chained** on the local RTX 2080 SUPER, one process per run. Cold evals add
minutes. Comfortably a same-day batch.

## Constraints restated

I1 (regime knobs only; option C stays behind its own pre-statement + ADR note); I6/I7/I8;
no success-definition / threshold / goal_thresh / episode_length / partner / prereg /
`SCHEMA_VERSION` changes; OM out of scope (#177); D-034 open but non-blocking (filter-off
posture). Stop points honoured: founder sign-off before launch; report after the verdict
before drafting Phase 4; the gate spike never launches from this plan.

## Cross-references

`../2026-06-10-success-static-probe/` (probe, options note, readiness note);
`../2026-06-10-regime-alignment/` (A1/A2 anchors; characterization);
ADR-007 §Stage 1b Rev 17; ADR-002 §Revision history 2026-06-10 (run_id fingerprint);
PR #217 (B1/B2 fixes); issues #214, #215.
