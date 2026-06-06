# Stage-1b AS — Third §4a Firing: Consultation Brief

**Date.** 2026-05-24
**Author.** Farhad Safaei
**Trigger.** `PHASE1_IMPLEMENTATION_PLAN.md` §4a runbook fired for the **third time** on 2026-05-24: the P1.05.9 single-cell AS-hetero smoke (PR #200; branch parent main at `117a012`) returned `success_rate = 0/20` after 8.27 GPU-h on RTX 2080 SUPER. The coupled bump (`total_frames` 100k → 1M; `happo.hidden_dim` 64 → 256) closed Surfaces 2 (training budget) and 7 (actor capacity) without lifting the gate.
**Triggering slice.** P1.05.9 (budget + actor-capacity remediation; merged PR #200).
**Prior firings.** 2026-05-20 (pre-widening; `2026-05-20/CONSULTATION_BRIEF.md`); 2026-05-21 (post-widening; `2026-05-21-post-widening/POSTWIDENING_INVESTIGATION.md`).
**Status of this brief.** Factual record (§1-§3, §5-§8) is the consultation's mechanical input; §4 founder prior is deliberately held off-disk per the discipline choice argued in §4 itself. Re-read seven days after first draft before any code is shipped or any candidate is acted on; if the rationale still holds and the §4 in-conversation prior has stabilised, proceed to §6.
**Solo-founder substitute discipline.** The runbook names "senior-advisor consultation" as the gate. **No advisor exists for this project** — this fact has been recorded since the 2026-05-20 brief (`CONSULTATION_BRIEF.md` §preamble + §5). The 2026-05-20 substitute preserved (a) writing the case in full before deciding, (b) imposing a 24-hour sleep on the conclusions, (c) recording the rationale in an ADR amendment when the remediation slice opens, and (d) cross-checks against named precedents (Liu 2024 RSS, COHERENT, Huriot–Sibai 2025) and upstream ManiSkill conventions. The third firing is qualitatively heavier than the first two — its surviving candidates (§3) are scientific-positioning decisions whose consequences cascade through Stage-2 and Stage-3 — so the substitute extends for this firing to (e) a **seven-day** sleep instead of 24 hours, (f) two new literature cross-checks specifically scoped to the surviving candidates (HARL HAPPO PickCube success reports for §3(a)/(e) framing; ManiSkill exploration baselines for §3(b)/(d) framing), and (g) a written devil's-advocate pass against whichever candidate the §4 in-conversation prior most strongly favours, recorded in the Rev 15+ ADR amendment alongside the chosen candidate. This is still weaker than a real advisor; it should be revisited if an advisor relationship becomes available before the consultation closes.

---

## §1 — What happened

P1.05.9 coupled two surviving hypotheses from the post-widening investigation
(2026-05-21) into a single falsification slice. The slice raised
`total_frames` from 100,000 to 1,000,000 in `stage1_pickplace.yaml` and
raised `happo.hidden_dim` from 64 to 256, holding every other lever from the
post-widening configuration constant. The coupling decision was logged in the
P1.05.9 design pass with the cost-vs-falsification trade-off recorded
verbatim — under the coupled bump, a clean `success_rate ≥ 20%` would close
both surfaces at once; a 0% result would close both at once with the cost of
not knowing which lever was the dominant constraint.

The RTX 2080 SUPER smoke completed in 8.27 GPU-hours and reported
`success_rate = 0/20`, `n_terminated = 0`, `mean_reward = 0.0337` averaged
over the eval window. The averaged reward sits ~0.006 above the
post-widening baseline of 0.0280, but the per-episode `max_reward` came down
from 0.0719 to 0.0494. The plateau is tighter, not lower, and the policy
still never grasps within the eval horizon.

The convergent fingerprint across all five recorded runs (2026-05-20 baseline,
post-widening smoke, A1 zero-action-partner, A2 safety-off, A5
partner-displaced, P1.05.9 coupled bump) is mechanical and consistent:

> The trained policy makes the panda reach toward the cube but never closes
> the gripper on it within the 50-step eval horizon, regardless of partner
> behaviour, safety-filter state, training budget, or actor capacity.

This fingerprint is what makes the third firing different in character from
the first two. The first firing identified instrumentation bugs (Surfaces 1
and 6). The second firing ruled out partner and safety surfaces (3 and 5).
The third firing has now ruled out the two surfaces (2 and 7) that the
literature would call the default scaling answers. The remaining surfaces
are no longer about instrumentation or fairness of the comparison — they are
about whether PickCube + HAPPO + 50-step horizon can produce grasp behaviour
at all under the project's current pre-registered protocol.

## §2 — Surfaces ruled out

The cumulative ruling-out across three firings:

| Surface | Description | Closed by | Evidence path |
|---|---|---|---|
| 1 | Rubber-stamp `mean_reward > -0.30` success rule | P1.05.8 / PR #185 | `2026-05-20/` |
| 2 | Training-budget shortfall (100k vs community 1M+) | P1.05.9 (this firing) | `2026-05-24-p1-05-9/` |
| 3 | Partner behaviour load-bearing | A1 zero-action partner (post-widening) | `2026-05-21-post-widening/` |
| 5 | Safety filter / partner displacement load-bearing | A2 safety-off + A5 partner-displaced | `2026-05-21-post-widening/` |
| 6 | Trainer obs synthesis (cube/goal/TCP/partner invisible) | P1.05.8 / PR #185 | `2026-05-20/` |
| 7 | Actor capacity / `happo.hidden_dim` underspecification | P1.05.9 (this firing) | `2026-05-24-p1-05-9/` |

Surfaces 2 and 7 were coupled in this firing; the §S.3 coupling rationale in
the P1.05.9 prompt records the founder accepting that closing both at once
makes the next iteration's pre-registration narrower, not wider. That trade
has now landed.

## §3 — Surviving candidates

The candidates below are the set the consultation should weigh. They are
ordered by what the post-widening fingerprint most directly implicates, not
by founder preference. Each carries an ADR §-reference where the relevant
discipline is already encoded.

**(a) Exploration / entropy collapse.** The reach-without-grasp pattern is
consistent with a policy that has converged to a deterministic non-grasping
optimum and is no longer exploring the action subspace that closes the
gripper. HAPPO's default entropy coefficient under
`configs/training/ego_aht_happo/stage1_pickplace.yaml` was not perturbed in
either P1.05.8 or P1.05.9. The ADR-002 §Rev 2026-05-20 cuda-major coupling
discipline does not constrain entropy-coefficient choices, so any
investigation here lives inside the implementation-detail allowance of
ADR-007 §Discipline rather than triggering a comparison-protocol amendment.

**(b) Reward-shaping discontinuity at `is_grasped`.** The PickCube reward
under the current `stage1_pickplace.yaml` is sparse across the grasp
boundary — the policy receives essentially no signal that distinguishes "TCP
near the cube" from "TCP near the cube *with the gripper closed on the
cube*" until the lift phase begins. The fingerprint is what one would expect
from a policy that has saturated on the reach component and cannot bridge
the credit-assignment gap to the grasp component. Acting here would require
a shaping amendment to the env definition. Any reward change MUST be
weighed against ADR-007 §Discipline (per-firing immutability) and
ADR-009 §Partner zoo (the comparison protocol bakes in a specific reward).

**(c) Stage-1b re-positioning away from PickCube.** This is the heaviest
move on the table. The pre-registered protocol pinned Stage-1b to PickCube;
re-positioning means re-cutting the `prereg-stage1-AS-2026-05-15` tag, which
the pre-registration discipline of ADR-007 §Discipline explicitly forbids
without an ADR amendment. The argument for considering it anyway is that
the convergent fingerprint suggests the task itself, not the method, is the
constraint — PickCube's 50-step horizon with sparse grasp reward is a known
hard exploration problem and the literature that reports success on it
typically uses BC seeds, demonstration-conditioned warm starts, or
auto-curricula, none of which are in the pre-registered comparison
protocol.

**(d) Eval-horizon mismatch.** The 50-step `max_episode_steps` in
`stage1_pickplace.yaml` may be shorter than the grasp+lift+place sequence
that the policy has learned to reach toward. If the policy reaches at step
30 and the eval horizon truncates at 50, the gripper may simply not have
been queried in the regime where grasp closure is plausible. This is a
falsifiable hypothesis with a near-zero-cost smoke (re-evaluate the latest
P1.05.9 checkpoint with horizon 100 or 200), and should run before any of
(a)–(c) is committed.

**(e) Curriculum / BC seed bootstrap.** A short pre-training pass with a
small set of scripted-grasp demonstrations or a 2-stage curriculum
(reach-only → reach+grasp) would convert the credit-assignment problem to a
fine-tuning problem. The community PickCube successes in HARL HAPPO and
neighbouring frameworks lean on bootstraps of this character. Adopting this
moves Stage-1b's pre-registered comparison protocol into "HAPPO + warm
start," which is a meaningful ADR-007 amendment but is a smaller move than
(c).

**(f) Method-vs-scaling diagnosis.** Even before the consultation chooses
among (a)–(e), the brief should record that further scaling (1M → 5M
frames, 256 → 512 hidden_dim) is unlikely to clear the grasp discontinuity.
The five-run fingerprint is structural, not budget-limited; the
P1.05.9 plateau came down at `max_reward` while ticking up at `mean_reward`,
which is the signature of a tighter local optimum, not an under-trained
policy. The question this surfaces for the consultation pass is whether
scaling is even a credible Stage-1b lever any more, or whether the project
should treat scaling as falsified for this task family and route through
(a)/(b)/(d)/(e) instead. This is not a candidate per se; it is the prior
that should frame how the consultation weighs the other five.

## §4 — Founder prior (deliberately held off-disk)

§4 is **deliberately left as a placeholder** on the immutable record. The
third firing's founder prior is held off-disk so it can be expressed
without pre-commitment during the live consultation pass — which, in the
solo-founder substitute discipline (preamble; cross-ref to 2026-05-20
`CONSULTATION_BRIEF.md` §5), is the seven-day sleep + literature
cross-checks + devil's-advocate pass over §3's surviving candidates. The
brief's factual record (§1-§3, §5-§8) is what the consultation works
from on paper; the prior lives in the founder's reasoning during the
sleep window and lands in the Rev 15+ ADR amendment (§6) alongside the
chosen candidate.

This placeholder choice is itself a discipline decision, not an
oversight. Pre-recording a founder prior on the three-way scientific
positioning the consultation has to weigh — entropy / exploration vs
reward shaping vs task-family re-positioning — would constrain the
consultation rather than inform it. The first two firings could carry
founder voice throughout the brief because their dominant defects
(instrumentation; partner+safety) were engineering questions where the
prior and the diagnosis lived on the same axis. The third firing
crosses into science-contract territory (§3(b)/(c)/(e) all touch the
comparison-protocol surface ADR-007 §Discipline guards); the
engineering-vs-science separation the 2026-05-20 brief named verbatim
in its §5 is preserved by deferring the prior to the consultation
moment.

What the consultation works from in §4's place: the §5 narrow question,
the §3 candidate set with its post-fingerprint ordering, the §3(f)
framing prior that further scaling is not credible, and the §6
amendment-scope ladder. The §4 placeholder is the seventh element — the
**absence** of a pre-committed prior is itself audit-trail-visible.

## §5 — Consultation question

The narrow question this consultation pass weighs (per the
solo-founder substitute discipline in the preamble: seven-day sleep +
literature cross-checks + devil's-advocate pass; no human advisor):

> Given the convergent fingerprint across five runs — reach without grasp,
> across partner / safety / budget / capacity ablations — which of
> candidates (a)–(e) should the next slice falsify, and which (if any)
> require an ADR-007 amendment that re-cuts the pre-registered
> comparison protocol versus living inside the implementation-detail
> allowance of ADR-007 §Discipline?

A subordinate question, surfaced by §3(f), is whether further scaling
(beyond 1M frames and `hidden_dim = 256`) should be treated as falsified
for this task family for the remainder of Phase 1, or whether the
consultation pass concludes that scaling is still a credible Stage-1b
lever the project has not yet exhausted.

## §6 — ADR-007 §Reversibility framing

ADR-007 §Reversibility records that the axis-selection decisions are
reversible by amendment if pre-registered evidence does not support them.
The third firing does not by itself invalidate the AS axis — what it
invalidates is the specific Stage-1b *instantiation* of that axis (PickCube
+ HAPPO + 50-step horizon + the pre-registered reward + the pre-registered
partner zoo). Any amendment authored after the consultation should be
careful to scope the reversal to the instantiation, not to the AS axis as
a whole; otherwise the audit chain reads as "the project abandoned the AS
axis" rather than "the project re-instantiated Stage-1b under the AS axis."

The amendment(s) the consultation is likely to require, in increasing order
of scope (Rev 14 is already taken by the P1.05.9 slice that fired this
brief; the next available revision number is Rev 15):

  — *Implementation-detail amendment* (candidates a, d): no comparison-
    protocol change; appended as ADR-007 Rev 15 with an Implementation
    Notes entry. No prereg tag rotation. ADR-007 §Discipline already
    permits this.
  — *Reward / curriculum amendment* (candidates b, e): comparison-protocol
    change. Requires ADR-007 Rev 15 explicitly amending the Stage-1b
    success rule and a new prereg tag (`prereg-stage1-AS-2026-05-XX`) cut
    against the new YAML. The prior `prereg-stage1-AS-2026-05-15` tag
    stays in place as historical record; the §20-pp pre-registered
    evidence claim narrows to the new tag's window. ADR-009 should be
    cross-referenced for the partner-zoo implications.
  — *Task-family re-positioning amendment* (candidate c): ADR-007
    Rev 15 + a new ADR (likely ADR-018) documenting the choice of
    replacement task family and the ADR-002 / ADR-009 compatibility
    review. The §20-pp claim is preserved if the replacement task is
    pre-registered before any results are looked at; it collapses if any
    replacement-task results are inspected before the prereg tag is cut.

## §7 — What does NOT happen before the consultation closes

  — No P1.05.9b, P1.05.9c, P1.05.10 slice opens.
  — No edits to `spikes/preregistration/*.yaml`.
  — No rotation of `prereg-stage1-AS-2026-05-15`.
  — No Stage-2 work resumes (CR / CM / PF / SA remain blocked).
  — The 2026-05-24-p1-05-9 investigation directory and its contents
    (SpikeRun JSON, launch.log, per-step JSONL, SHA256SUMS.txt) are
    immutable per ADR-007 §Discipline.

## §8 — Audit artifacts

Co-located in this directory:

  — `spike_as_hetero_p1_05_9_FAILED.json` — full run config + result summary
    (SpikeRun envelope; 1 seed × AS-hetero × 20 eval episodes;
    `success_rate = 0/20`).
  — `launch.log` — 8.27 GPU-hour smoke driver stdout on RTX 2080 SUPER
    (driver 580.159.03; torch 2.11.0+cu128; merged commit `117a012`).
  — `f7371e3f5c55785d.jsonl` — per-rollout training event stream from
    `concerto.training.ego_aht.train` for `run_id = f7371e3f5c55785d`;
    carries `training_start`, per-rollout `safety_telemetry` +
    `rollout_update`, `checkpoint_saved` (×100), and `training_end` /
    `safety_telemetry_final` events (3031 lines, 1.7 MB).
  — `SHA256SUMS.txt` — content hashes for the above plus this brief.
  — `THIRD_FIRING.md` — this brief.

Once the brief is finalised and the directory is signed-and-committed, the
audit chain for Stage-1b spans three immutable directories
(`2026-05-20/`, `2026-05-21-post-widening/`, `2026-05-24-p1-05-9/`) — the
mechanical record the consultation will work from.
