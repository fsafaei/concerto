# Tier-1 reading note — Construct validity in cooperative-AI evaluation

**Note id.** `notes/tier1/construct_validity_in_cooperative_evaluation.md`
**Reader.** Farhad Safaei
**Committed.** 2026-07-05 (previously maintained in the local reading corpus; committed to satisfy the ADR-INDEX §Locking-rule citation duty for ADR-026's promotion)
**Cited by.** ADR-026 §Evidence basis; ADR-018 §Evidence basis; ADR-027 §Evidence basis.

## Scope of this note

The claim this note grounds: *a heterogeneity manipulation is informative
about cooperation only if it is coupled to the task outcome through the
cooperation the task demands* (the coupling-validity criterion,
ADR-026 §Decision 1), and the operational discipline around it
(positive controls, held-out partners, interval-based reporting). Each
source below predates the 2026-06-11 Stage-1 AS verdict, which is what
lets ADR-026 ground the criterion in prior methodology rather than in
the unwelcome result it reinterprets (ADR-026 §Risks #1).

Verification status: every work below is a published, retrievable
source; the claims attributed to each are verified at the level of the
paper's stated contribution and evaluation design (the level at which
ADR-026 cites them), not by page-anchored quotation. Where a claim is
paraphrase, it is marked as such.

## Sources

### 1. Cronbach & Meehl 1955 — the origin of construct validity

Cronbach, L. J., and Meehl, P. E. "Construct Validity in Psychological
Tests." *Psychological Bulletin* 52(4), 1955, pp. 281–302.

What it establishes (paraphrase): a test is construct-valid only
relative to a *nomological network* — the lawful relations the
construct is supposed to enter. A measurement procedure that produces
stable numbers can still fail construct validity if the manipulation
does not connect to the construct through those relations.

Bearing on the decision: the Stage-1 AS measurement produced a stable,
tight-interval number (−11 pp) on a task whose success predicate reads
only the ego agent. The number is real; the construct link
(heterogeneity → *cooperation* → outcome) is absent. This is exactly
the Cronbach–Meehl distinction between measurement reliability and
construct validity, and it is why ADR-026 reinterprets rather than
re-runs the verdict.

### 2. Messick 1995 — validity is a property of inferences, not instruments

Messick, S. "Validity of Psychological Assessment: Validation of
Inferences from Persons' Responses and Performances as Scientific
Inquiry into Score Meaning." *American Psychologist* 50(9), 1995,
pp. 741–749.

What it establishes (paraphrase): validity attaches to the *inference
drawn from a score*, not to the instrument; the same score can support
one inference and not another. Two named threats are
construct-underrepresentation and construct-irrelevant variance.

Bearing on the decision: the AS archive stays immutable and can
support rig-level inferences (Stage 1 as rig-validation, ADR-026
§Decision 5) while being invalid for the cooperation inference. The
incidental-disturbance channel through which embodiment entered the
Stage-1 outcome is construct-irrelevant variance in Messick's sense.

### 3. Stone, Kaminka, Kraus & Rosenschein 2010 — the ad-hoc-teamwork problem statement

Stone, P., Kaminka, G. A., Kraus, S., and Rosenschein, J. S. "Ad Hoc
Autonomous Agent Teams: Collaboration without Pre-Coordination."
*Proceedings of the Twenty-Fourth AAAI Conference on Artificial
Intelligence (AAAI 2010)*, Atlanta.

What it establishes (paraphrase): the ad-hoc-teamwork challenge is
defined as creating an agent that cooperates with previously unknown
teammates, and its proposed evaluation is explicitly *cooperative*: the
measure of an ad-hoc agent is team performance when embedded with
teammates it was not developed with, over tasks the team must
accomplish jointly. The teammates are not under the designer's control
and are not transparent to the agent.

Bearing on the decision: two commitments follow. (i) An AHT evaluation
task must actually demand teamwork — a task the ego can solve alone
cannot separate an ad-hoc cooperator from a competent solo agent; this
is the coupling-validity criterion in its original habitat. (ii) The
teammate enters the evaluation as a behavioural black box — the basis
of the black-box partner contract (ADR-018), under which the ego may
observe the partner's *behaviour* (pose, actions in the world) but
never its policy internals.

### 4. Leibo et al. 2021 — Melting Pot's held-out-partner evaluation design

Leibo, J. Z., Dueñez-Guzman, E. A., Vezhnevets, A. S., Agapiou, J. P.,
Sunehag, P., Koster, R., Matyas, J., Beattie, C., Mordatch, I., and
Graepel, T. "Scalable Evaluation of Multi-Agent Reinforcement Learning
with Melting Pot." *Proceedings of the 38th International Conference on
Machine Learning (ICML 2021)*, PMLR 139.

What it establishes (paraphrase): a cooperative-MARL evaluation suite
where the unit under test (the focal population) is scored against
*held-out background populations* it never trained with, in test
scenarios constructed so that the background agents' behaviour is what
the focal agents must adapt to. The test set is defined by the
partners, not only by the environment; generalisation claims are made
per-scenario, not as one scalar.

Bearing on the decision: this is the closest published template for
CHAMBER-Bench's shape — frozen, versioned partner sets as the test
instrument (ADR-009 as amended; ADR-027 §Decision), per-task reporting
without a scalar composite, and the discipline that the partner
population is part of the benchmark's identity (hence partner-set
versioning in ADR-027/028).

### 5. Raji et al. 2021 — benchmark construct validity

Raji, I. D., Bender, E. M., Paullada, A., Denton, E., and Hanna, A.
"AI and the Everything in the Whole Wide World Benchmark." *NeurIPS
2021, Datasets and Benchmarks Track.*

What it establishes (paraphrase): benchmarks routinely fail construct
validity — the benchmark measures something narrower than, or different
from, the capability named in its framing — and the failure is a
property of task construction, diagnosable from the task itself, not
from the leaderboard numbers it produces.

Bearing on the decision: authorizes diagnosing the Stage-1 defect from
the environment source (ego-only success predicate; non-participating
partner) independent of the measured outcome, which is what makes the
ADR-026 reinterpretation principled rather than outcome-driven. Also
the standing reason ADR-027 refuses a scalar composite leaderboard for
v1.0.

### 6. Gorsane et al. 2022 — standardised cooperative-MARL evaluation protocol

Gorsane, R., Mahjoub, O., de Kock, R. J., Dubb, R., Singh, S., and
Pretorius, A. "Towards a Standardised Performance Evaluation Protocol
for Cooperative MARL." *NeurIPS 2022.*

What it establishes (paraphrase): a documented, pre-committed
evaluation protocol — fixed evaluation parameters, controlled sources
of variance, complete reporting across seeds — is the corrective for
the reproducibility and cherry-picking pathologies the paper audits in
the cooperative-MARL literature.

Bearing on the decision: grounds the pre-registration discipline that
ADR-026 §Decision 2 makes binding (the coupling positive-control and
the pre-committed null rule) and that ADR-027 carries into the
admission protocol (A1–A3 thresholds preregistered before measurement).

### 7. Agarwal et al. 2021 — interval estimates over point estimates

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., and
Bellemare, M. G. "Deep Reinforcement Learning at the Edge of the
Statistical Precipice." *NeurIPS 2021* (the `rliable` methodology).

What it establishes (paraphrase): point estimates over few runs
routinely misrank methods; the recommended practice is interquartile
mean (IQM) with stratified-bootstrap interval estimates, reported as
intervals, with performance profiles over aggregate scalars.

Bearing on the decision: the source of ADR-026 §Validation criteria's
forward statistics (IQM secondary to the mean-delta gate,
bootstrap CIs over seed clusters) and ADR-027 §Reporting rules (IQM +
95 % paired-cluster-bootstrap CIs, no single-seed claims).

## Synthesis (the passage ADR-026 leans on)

The seven sources triangulate one methodological fact: *what a
cooperative benchmark measures is fixed by how the task couples the
agents, and what its numbers license is fixed by protocol committed
before measurement.* Construct validity is diagnosable from task
construction (1, 2, 5); a cooperation claim requires a task the agent
cannot solve alone and a partner it cannot see inside (3, 4); and the
reported effect must survive preregistered, interval-based statistics
over held-out partners (4, 6, 7). The coupling-validity criterion is
the first half of this fact applied to heterogeneity axes; the
admission protocol and reporting rules of ADR-027 are the second half.

## Open questions carried forward

- Melting Pot's background populations are *trained* artefacts; CHAMBER
  v1.0's scripted stratum is hand-built. Whether scripted partners
  under-represent the behavioural diversity a learned background
  population provides is an empirical question for the partner-set
  work under ADR-009 as amended.
- None of the sources settles how to aggregate *across* tasks when the
  per-task validity statuses differ; ADR-027 answers by suspending the
  aggregate (HRS computes only over `validated` cells), which is a
  project decision, not a literature result.
