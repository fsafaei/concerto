# Consultation brief — Stage-1b AS gate verdict (sign-reversed) — DRAFT

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** DRAFT for founder review; the founder sends it. Audience: the senior advisor.
Everything cited is on `main` or in sealed archives; nothing in this brief re-runs, re-frames,
or re-estimates the recorded verdict (exit 5 stands regardless of any follow-up).

---

## §1 — The measured verdict, no spin

The first gate-scale AS spike on the fully-repaired rig returned **exit 5** (gate FAIL) with
the pre-registered gap **sign-reversed**: the hypothesis was a homogeneity advantage
≥ +20 pp on the 95 % CI lower bound; the measurement is

- pooled success: **HOMO 1/100 vs HETERO 12/100** (gap ≈ **−11 pp**, hetero better);
- per-seed deltas (homo − hetero): −15, −35, 0, −5, 0 pp;
- recorded gate read (prereg estimator, IQM over the paired-cluster bootstrap):
  `ci_low_pp = −2.00, ci_high_pp = 0.00`;
- both conditions are *structurally healthy*: cold grasp AND place are 30/30 on **all ten
  cells** (the W-A watch item did not fire); the entire difference lives in the
  settle-conversion step of the success predicate (W-B occupancy decomposition).

**The evidence chain behind the rig (one paragraph, pointers).** Episode-horizon enforcement
(ADR-007 Rev 15, PR #206) and reset-state init (Rev 16, PR #210) fixed two confirmed env
defects; the training-time CBF-QP's uncalibrated ±0.1 action clamp was isolated as the
dominant consolidation blocker by a single-variable filter-off arm (A3;
`2026-06-10-regime-alignment/`); regime alignment to field practice (1,024 envs × 20M × γ=0.8;
Rev 17, PR #213) produced 30/30 cold grasp+place on 3/3 seeds with zero levers; a γ-scan
(`2026-06-11-gamma-scan/`, 7 runs) showed no discount bridges the static-conjunct valley; a
potential-based settle term in exact Ng–Harada–Russell form (Rev 18, PR #219; eval always
unshaped) qualified at α=0.5 under a frozen rule (`2026-06-11-pbrs-settle/`); the gate then
ran the prereg protocol verbatim, condition-symmetrically, 10/10 cells clean
(`spikes/results/stage1-AS-2026-06-11/GATE_VERDICT_REPORT_2026-06-11.md`).

## §2 — Two candidate readings (conditioned on the operationalization facts)

The companion document `AS_HOMO_OPERATIONALIZATION_2026-06-11.md` (code-cited) establishes:
**(i)** the homo cell as executed is *not* the prereg-notes baseline — the notes describe
shared-parameter MAPPO (two co-learning pandas); what ran is the identical ego-AHT-HAPPO
learner vs a frozen scripted second panda — "the hetero learning problem with a panda where
the fetch stands"; **(ii)** a known, ticketed heuristic mis-port (`state[:2]` read as planar
xy but actually the first two joint positions; P1.05.10) acts **asymmetrically by
condition**: the hetero fetch partner is an empirical statue, while the homo panda partner's
arm is *perpetually moving through the shared workspace every episode* (measured:
qpos[1] 0.393→0.116 over 30 steps, |qvel| 0.09–0.27 throughout).

- **Reading (a) — operationalization artifact.** The homo baseline loses for reasons
  orthogonal to embodiment: it is not the prereg-described co-learning baseline, AND its
  partner was uniquely perturbed by the mis-port. A further selection-pressure note: every
  regime and lever decision in the campaign (horizon, reset, regime, γ, α) was *validated on
  hetero cells only* — settings were applied symmetrically, but the selection pressure was
  not; α\*=0.5 was chosen because it converts *hetero* settles.
- **Reading (b) — genuine signal.** A second 7-DOF arm sharing the workspace degrades
  settle-phase convertibility in ways a distant differential-drive base does not: the ego
  must hold still while another long arm looms over the table (even a quasi-static one), and
  homo's W-B profile (almost never *attempts* settling; perfect placement retention on its
  rare settles) is consistent with a policy that learned settling is risky near the partner
  arm. Honest caveats for (b): the frozen-partner/C2 falsification exists **only for
  hetero** (the success-static probe never ran a homo control), and (b) must be trusted
  *through* both (i) and (ii).

The two readings are not exclusive — (a) can fully explain the homo floor while (b) explains
the residual hetero seed-variance — but they imply different next experiments (§5.ii).

## §3 — The estimator question (W-C fired exactly as pre-stated)

The prereg pins `estimator: iqm_success_rate`; the gate module's pre-commitment
(`chamber/cli/_spike_next_stage.py`, module docstring, on `main` since T5b.1) reads,
verbatim: *"For bimodal binary delta data — where the per-pair delta is 0/1 rather than a
smooth fraction — IQM can pin to 0 even when the mean is non-zero because the middle 50 % of
sorted resampled values collapse to the majority value. Future work (ADR-008 amendment) may
add a `--gate-aggregator {iqm,mean}` flag if real-spike borderline data makes the choice
load-bearing; `BootstrapCI` already carries both the IQM and the mean."*

On this data: the IQM read is **[−2.00, 0.00]** while the carried mean point estimate is
**−11.00 pp** — the pinning is no longer hypothetical. Note ADR-008's §Decision *fallback
rules* concern HRS **bundle composition** (Option D → B/A by surviving axes), not the
estimator; an estimator change is exactly the "ADR-008 amendment" the module names, and the
prereg pin means it cannot apply to AS retroactively. **The question for you is
forward-looking only: which pre-committed estimator should govern the OM prereg and any
freshly-preregistered AS re-measurement — IQM as-is, mean, or both-reported-gate-on-one?**
This brief proposes nothing post hoc for the recorded AS result.

## §4 — The OM question

OM compares observation-modality heterogeneity (vision-only vs vision+FT+proprio) on the
**same agent tuple in both conditions** — no homo-operationalization confound of either §2
kind. It has never fired: blocked by issue #177 (the trainer reads a flat `state` key; the
OM wrapper chain intentionally does not synthesise it for vision-only — a vision-head trainer
is required; ADR-007 Rev 9 records the deferral and that shipping OM without it would either
corrupt the science or crash). The pre-commitments on record: the gate needs **one** axis
(ADR-007 §Validation criteria: "≥1 of {AS, OM} ≥ 20 pp"); Rev 9 sequences P1.05.6 (the
vision head) before any Phase-1 closure; and OM now inherits a proven regime at ≈ 5.5 h per
10-cell axis on the local GPU (the A100 budget question is dissolved). A separate unblock
assessment (Phase 3 of the founder's processing plan) will scope #177's smallest fix.

## §5 — The specific asks

1. **A third reading?** Do you see an interpretation of {homo 1/100, hetero 12/100, both
   conditions 30/30 on grasp+place, conversion-only difference} that is neither §2(a) nor
   §2(b)?
2. **Highest evidential value per hour:** (A) unblock OM (#177) and run the OM gate at the
   proven regime (~5.5 h once unblocked; no operationalization confound); (B) the
   exploratory homo cell with the partner asymmetry removed (3 seeds ≈ 1.7 h; directly
   separates §2(a) from §2(b); EXPLORATORY — cannot change the recorded verdict); (C) both,
   and in which order?
3. **The estimator:** does §3 warrant the named ADR-008 amendment (gate aggregator choice)
   *before* the OM prereg is cut, so OM's estimator is chosen with eyes open rather than
   inherited?
4. **External validity:** any concern with eventually publishing the sign-reversed
   pre-registered AS result alongside whatever positive evidence accrues — i.e., is
   "pre-registered hypothesis refuted, possibly for operationalization reasons, with the
   full repair history documented" a strength or a liability in your view, and how would
   you frame it?

## Pointers (all on `main`)

`spikes/results/stage1-AS-2026-06-11/` (GATE_VERDICT_REPORT + sealed archive);
`AS_HOMO_OPERATIONALIZATION_2026-06-11.md` (this directory); the campaign chain
`2026-06-10-regime-alignment/` → `2026-06-10-success-static-probe/` →
`2026-06-11-gamma-scan/` → `2026-06-11-pbrs-settle/` → `2026-06-11-gate-regime-spec/`;
ADR-007 §Validation criteria + §Stage 1b Revs 9, 12–18; ADR-008 §Decision + the
`_spike_next_stage.py` pre-commitment; ADR-009 §Decision; issues #177, #227, P1.05.10;
`spikes/preregistration/{AS,OM}.yaml` @ their 2026-05-15 tags.
