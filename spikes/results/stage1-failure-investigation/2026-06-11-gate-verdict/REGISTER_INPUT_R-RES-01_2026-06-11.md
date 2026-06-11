# Register input — R-RES-01 after the AS gate verdict (kit input; NOT a register edit)

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** Staged kit input for the planning kit's risk register (kit not mounted on this
host; transfer per the established D-034 pattern). **This note lays out the options and
decides nothing** — the register is the founder's document.

## The facts against the trigger

R-RES-01's early-warning trigger reads: *a post-rig-validation firing closing < 20 pp on
**both** axes.* As of 2026-06-11:

- **AS:** fired at gate scale on a rig with every known defect fixed and consolidation
  demonstrated; closed at `ci_low_pp = −2.00` (exit 5). The post-rig-validation qualifier is
  satisfied for AS — the repair chain (Rev 15/16, clamp isolation, Rev 17/18) and the W-A
  read (grasp+place 30/30 on all ten cells) are on `main`.
- **OM:** has never fired (blocked by #177; unblock assessed at 2–4 founder-days plus
  rendering-bound gate costs — `OM_UNBLOCK_ASSESSMENT_2026-06-11.md`).

**The both-axes condition is therefore not evaluable today.**

## Two defensible register treatments (unchosen)

1. **Hold un-armed pending OM.** Strict reading of the trigger text: one axis closed, one
   axis unfired ⇒ the trigger condition is not met; the register entry gains a dated
   annotation ("AS closed <20 pp post-validation 2026-06-11; trigger evaluable when OM
   fires") and nothing else changes. Keeps the register's semantics exact; defers the
   early-warning posture until the OM gate, which is weeks away at the assessed effort.
2. **Annotate AS-closed-now.** Pragmatic reading of the trigger's *purpose* (early warning
   that the heterogeneity-penalty thesis may not survive contact at gate scale): half the
   trigger has fired in the worst available direction, so the register marks R-RES-01
   "partially armed — AS leg closed," and the pre-committed responses (consultation
   discipline, §4a runbook, ADR-008 fallback review) begin now rather than after OM. This
   is materially what is already happening (the consultation brief exists); treatment 2
   makes the register match reality.

Operational note: under either treatment, nothing about the recorded AS verdict or the
prereg'd protocols changes; the difference is purely when the register's response machinery
is considered "engaged."

## The wording flag (for the next premortem refresh; not for now)

R-RES-01 anticipated *closing* — a too-small gap. What the AS gate measured is a
**sign-reversed** gap (hetero outperforms homo, −11 pp pooled), with a live candidate
explanation that the homo baseline was mis-operationalized
(`AS_HOMO_OPERATIONALIZATION_2026-06-11.md`). The register's risk taxonomy has no entry for
"the baseline condition, not the heterogeneous condition, fails to perform" — which is a
different failure mode with different responses (operationalization audit, baseline
re-preregistration) than the anticipated "heterogeneity penalty too small to measure." Flag
for the next premortem refresh: either widen R-RES-01's wording or add a sibling entry for
baseline-operationalization risk. No wording change is proposed in this note.

## Cross-references

`spikes/results/stage1-AS-2026-06-11/GATE_VERDICT_REPORT_2026-06-11.md`;
`CONSULTATION_BRIEF_GATE_VERDICT_2026-06-11.md`; `AS_HOMO_OPERATIONALIZATION_2026-06-11.md`;
`OM_UNBLOCK_ASSESSMENT_2026-06-11.md`; ADR-007 §Validation criteria; issue #177.
