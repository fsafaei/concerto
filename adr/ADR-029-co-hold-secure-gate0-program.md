# ADR-029: co_hold_secure Gate-0 program — the second-discriminating-task pivot

**Status.** RFC (2026-07-16) — the founder flips this to Accepted (or amends) at PR review, per §7 of the ratified pivot decision.
**Authors.** Farhad Safaei
**Reviewers.** _solo lock per ADR-INDEX working policy_
**Tags.** v1.1 benchmark scope; depends on ADR-026, ADR-027, ADR-028; invariants I1/I3/I8/I9.

## Context

CHAMBER-Bench v1.0 shipped with exactly **one** discriminating cooperation
task (co-carry). ADR-027 names "benchmark of one" as the top external-review
risk, and the v1.1 goal is a second discriminating task. The intended vehicle
was a learned-ego ladder on the admitted `handover_place` task; the Slice-0
oracle-headroom probe (PR #309, rule pre-stated at tag
`probe-handover-ladder-slice0-2026-07-16`) returned **NO-GO** by the committed
rule:

- **Headroom 0.000** — a per-state oracle search of the entire ego action
  space through the real env scores exactly what the scripted reference
  scores on both canonical cells (0.548 and 0.128). The committed REF row is
  the task *ceiling*, not the scripted rule's shortfall.
- **Learnable structure 0.000** — the oracle never beats the best fixed
  policy; the one decision that matters is already a single threshold on an
  observed quantity, and the scripted rule sits on the optimal threshold.
- **Every failure is budget-infeasible by construction** (angular error
  beyond the wrist correction; the takt budget blocks the re-grasp), so no
  action exists that succeeds on the failing draws.

`handover_place` itself is untouched — it stays admitted (A1–A3 passed,
Tier 2) and its scripted leaderboard rows stand. What died is only the
learned-ladder extension on that vehicle. A second discriminating task
therefore needs a different task, and the registry already carries the
carded candidate: `co_hold_secure@v0` (Tier 3, CANDIDATE, `env_factory=None`).

The nearest prior art is the co-insert closure
(`spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md`): a competent,
reusable two-robot hold-and-insert rig that honest-closed at a **geometric
tilt-wedge** — seating a 16 mm peg to the 38 mm threshold at 0.5 mm-per-side
clearance requires the relative peg–bore tilt held below ~0.7°, while the
insertion contact itself cocks the peg to ~0.9–2.8°. The seatable region and
the achievable-control region did not overlap, at any frozen clearance.

## Considered alternatives

1. **(A) Open the co_hold_secure Gate-0 program** — build the carded v1.1
   flagship candidate (holder + securing operator under continuous contact)
   by adapting the banked co-insert rig, staged so the cheap kill switches
   fire before any heavy spend. **Chosen.**
2. **(B) A contact-rich handover variant** — redesign `handover_place` so the
   coupling is a continuous force interaction. Parked as the named fallback
   *vehicle*: the current handover env is pure-Python kinematic with nothing
   reusable for contact, and re-adding contact to force discrimination risks
   constructing difficulty for the method (the task-selection inversion).
3. **(C) No second task now** — ship the benchmark paper on co-carry + the
   admitted scripted handover + the honest NO-GO discipline narrative.
   Pre-named as the fallback *posture* if this program's precheck or Gate-0
   fails; it concedes the benchmark-of-one risk without attempting the
   mitigation the registry already carries.
4. **(D) An observation-modality axis on co-carry** — declined: it adds an
   axis, not a task, and does not answer "benchmark of one".

## Decision

Open the **co_hold_secure Gate-0 program** (option A), with option C
pre-named as the fallback posture and option B parked as the fallback
vehicle. The program is staged behind pre-stated kill switches:

1. **PR-A (this ADR's slice): decision pack + rig adaptation + engineering
   precheck.** `CoHoldSecureEnv` is adapted from the banked co-insert rig
   (`chamber.envs.coinsert` + `chamber.partners.coinsert_impedance`), the
   task spec is wired (`co_hold_secure@v1`, still Tier 3 / CANDIDATE — tier
   moves only via the admission protocol), and a **pre-stated,
   non-registered** solvability/coupling precheck ends the slice with a
   PROCEED or STOP verdict (rule committed before any measured episode;
   archive under `spikes/results/coholdsecure/`). No prereg tag is created
   or rotated in PR-A.
2. **Founder gate.** PROCEED ⇒ the founder signs the Gate-0
   pre-registration tag (thresholds locked before any registered run).
   STOP ⇒ fall back to option C; the closure paragraph is recorded in this
   ADR's revision history and the total loss is one slice.
3. **PR-B: registered Gate-0** — the coupling-validity + solvability verdict
   under the signed tag (the ADR-027 admission instruments A1–A4 apply at
   admission time; the specific public industrial-tolerance citation duty
   attaches at Gate-0 prereg time).
4. **Only after Gate-0 passes** is a learned-ladder program plan for
   co_hold_secure written and separately ratified. It would reuse the
   co-carry training surface (same SAPIEN + HARL/HAPPO path), which is why
   this vehicle — unlike handover — needs no from-scratch RL enablement.

**The wedge-inverted design rule (the load-bearing task-design decision).**
The co-insert failure geometry is inverted into a constructive constraint:
*choose engagement depth and per-side clearance so the seatable region
contains the achievable-control region with margin.* Operationally, with the
empirically-consistent two-point wedge law

```
theta_wedge ≈ arctan(per_side_clearance / engagement_depth)
```

(the S2 archives record seatable tilt < ~0.7° at 0.5 mm/side and 38 mm depth;
arctan(0.5/38) = 0.75°), and the measured achievable-control ceiling of
**2.8°** (the upper end of the S2 ~0.9–2.8° contact-cocking band), the
design window requires

```
theta_wedge(full engagement) ≥ 2 × 2.8° = 5.6°
```

The v1 task geometry sits inside that window: engagement depth **10 mm**,
per-side clearance band **{1.5, 2.5, 3.5} mm** (wedge-limit tilts 8.5° /
14.0° / 19.3° = 3.0× / 5.0× / 6.9× the ceiling), a **≥ 2 mm, 45°** lead-in
chamfer, and a **40 N detent (seat-click) resistance over the final 2 mm** of
travel as the securing process load. The securing axis is **deliberately
non-vertical (60° from vertical)**: a vertical push onto a passively
supported part is braced by the support, which would defeat the
two-robot-necessity control by construction, while at 60° the detent load's
lateral component exceeds what stand friction can react for any plausible
friction coefficient (F·sin60° > μ·(F·cos60° + m·g) for μ < ~1.4) — and the
tilt keeps both arms' press path inside a joint-limit-comfortable
configuration band (solved on the ADR-004 FK chain; margins ≥ 0.28 rad along
the full press).

**The A2 posture (decided here, not silently resolved).** A world-welded
part makes securing trivially easy, so the task must state how it meets the
ADR-027 A2 falsifier ("if a passive fixture plus a single robot matches the
team at the same tolerances, the task is not admitted"):

- **Route (i) — application-grounded (adopted for v1):** the fixtureless
  setting is the industrial anchor (high-mix cells where a dedicated
  fixture per part does not exist — the card's sourced anchor class:
  fixtureless welding / machining / fastening), and the C-fixture control is
  run and reported honestly beside the team cells rather than hidden.
- **Route (ii) — in-episode multi-pose securing sequence (designed as a
  dormant hook):** the holder must re-present the part between securing
  operations, so a static fixture at one pose cannot serve. The env carries
  a pose-sequence parameter defaulting to length 1; enabling it is a task
  change (version bump + new ADR review) if route (i) proves insufficient.

**The B-BLIND doctrine (Slice-0 disposition, recorded as ladder-composition
doctrine).** On co-carry, "blind" masks the partner's state while the ego
keeps its own proprioception; on handover-place the ego's entire
coupling-relevant input *is* its own perception of the presented pose, so
masking it is sensor-blinding, not partner-blinding. **Any future ladder on
any task must pass an own-channel / partner-channel definability check
before a B-BLIND cell is preregistered.**

## Rationale

- The registry and task card already carry co_hold_secure as the v1.1
  flagship candidate with the co-insert correction baked in ("a fastening or
  connector seat, never a zero-clearance peg"); choosing it is consistent
  with the recorded escalation intent, but it is registered as a fresh
  decision — the handover-place Gate-0 escalation clause was written for a
  WASHOUT verdict that never fired, not for a learned-ladder headroom NO-GO.
- The co-insert closure banked a competent rig **and the geometry of its
  failure**; the wedge law is the first task-design input in this project
  that is measured rather than assumed.
- The decouple ablation (PR #298, `COOPERATION_CONTINGENT`) validated the
  C2-style limp-partner instrument as the right coupling-liveness control,
  so coupling-liveness is checkable in the cheap precheck rather than
  discovered at admission.
- The admission protocol A1–A4 is executable code (`chamber.evaluation.
  admission`, #307/#308), so the spec can name the exact gates the task will
  later face.
- The honest counter-case is stated, not buried: co-insert also looked
  well-grounded and closed at a HARD_STOP after real spend. Two prior arcs
  (co-carry too forgiving, co-insert too unforgiving) both missed the
  solvable ∩ coupling-valid middle. The response is the staging — PR-A ends
  at a precheck whose failure costs days, and option C is a respectable
  landing.

## Evidence basis (links to reading notes)

- `spikes/results/handover-ladder-probe-2026-07-16/` (PROBE_REPORT.md;
  PR #309, merged `bdd869f`) — the Slice-0 NO-GO record.
- `spikes/results/coinsert/COINSERT_CLOSURE_2026-06-24.md` + the S2 archives
  (`spikes/results/coinsert/s2/…`) — the tilt-wedge numbers this ADR inverts:
  seatable < ~0.7° at 0.5 mm/side and 38 mm; achievable ~0.9–2.8°; the
  fixed-link attach collapsing the 573–1306 N `create_drive` preload to
  ~0.2 N (banked finding 1).
- PR #298 (`cocarry_decouple` ablation, `COOPERATION_CONTINGENT`) — the
  limp-partner (zero-action, coupling-intact) instrument precedent.
- ADR-026 §Decision (coupling-validity criterion), ADR-027 §Admission
  (A1–A4), ADR-028 (prereg document-form schema PR-B will use).

## Consequences

- `co_hold_secure` moves from spec-only (`@v0`, `env_factory=None`) to
  runnable (`@v1`, `chamber.envs.co_hold_secure.make_co_hold_secure_env`)
  while **staying Tier 3 / CANDIDATE** — admission status changes only via
  the ADR-027 protocol with a signed prereg.
- The stress channel is pinned as the **holder workpiece-wrench under
  process load** (the friction-inclusive `get_link_incoming_joint_forces`
  instrument generalised from co-insert / co-carry).
- The success predicate extends the co-insert conjunction with a
  **pose_held** conjunct (the part stays within the declared pose tolerance
  under the securing load) — the industrial requirement that makes a limp
  holder fail honestly rather than cosmetically.
- A PROCEED verdict obliges nothing except enabling the founder gate; a
  STOP verdict closes the program at option C with a one-paragraph record.

## Risks and mitigations

- **Another honest null after env spend (the co-insert precedent).**
  Mitigated by ordering: the wedge-derived design window is checked *before*
  the rig is built, and the precheck (solvability + coupling-liveness +
  stress-channel liveness) fires before any registered run or training
  spend.
- **The A2 tension is real** — C-fixture is expected to succeed at v0
  tolerances. Route (i) carries it as an honestly-reported control with the
  application-grounded argument; route (ii) is the designed escape hatch.
  The admission call itself belongs to PR-B and the ADR-027 instruments.
- **Detent instrument realism.** The detent is an analytic force pair
  applied inside the physics loop (action–reaction on plug and part), not a
  geometric interference — deliberately, because the S2 archives showed
  constraint-fidelity artifacts (hundreds-to-1300 N preloads) when rigid
  geometry is used to emulate what is physically a compliant feature. The
  precheck's P4 monotonicity + boundedness bound is the instrument check.
- **Tuning-to-pass.** Controller gains may be tuned freely during PR-A
  bring-up (nothing is frozen yet), but the *geometry* window is fixed by
  the derivation above, the precheck bounds are committed before any
  measured episode, and the Gate-0 thresholds are locked by the founder-
  signed tag before any registered run.

## Reversibility

Fully reversible at every stage: STOP at the precheck (fall back to option
C, one paragraph here), founder declines the Gate-0 tag, or Gate-0 fails in
PR-B. The env and spec wiring are additive (no Stage-1/Phase-1 surface is
touched); reverting to spec-only is a version bump back.

## Validation criteria

- The §1-derived design window is enforced in code
  (`chamber.envs.co_hold_secure.cohold_wedge_limit_deg` and the design-
  window check) and pinned by Tier-1 tests.
- The precheck archive (`spikes/results/coholdsecure/precheck-…/`) shows the
  pre-statement committed **before** the results commit, with the verdict
  computed by the pre-stated rule from the committed JSON.
- `chamber.tasks.make("co_hold_secure")` returns the env; the
  NotImplementedError placeholder path is gone; the rendered task card shows
  Gate-0-in-progress.
- No prereg tag, no `SCHEMA_VERSION` / `PREREG_SCHEMA_VERSION` bump, no
  tier or leaderboard change anywhere in PR-A.

## Open questions deferred to a later ADR

- The Gate-0 pre-registration itself (thresholds, n, decision rule, the
  sourced public tolerance citation) — PR-B, under the founder-signed tag.
- The learned-ladder program plan (the Slice 1–4 analogue) — only after
  Gate-0 passes, separately ratified.
- Whether route (ii) (multi-pose securing) is ever enabled — only if route
  (i)'s application-grounded A2 posture proves insufficient at admission
  review.
