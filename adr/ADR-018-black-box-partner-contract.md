# ADR-018: Realism constraint — the black-box partner contract

**Status.** Accepted (2026-07-05)
**Authors.** Farhad Safaei
**Reviewers.** _solo lock per ADR-INDEX working policy_
**Tags.** v0.2 §3.5, §6.1; realism constraint; partner zoo; AHT contract.

## Context

The black-box partner contract has been enforced in code since Phase 0
and cited by pre-registrations, spike reports, and ADR revision
histories as if a governing ADR existed — the 018 slot was reserved for
it at the ADR-026 numbering note but never filled. This ADR formalizes
what is already load-bearing, so every citation of the contract
resolves to a real decision record.

The contract answers a realism question: what may a robot know about a
teammate it was never trained with? The ad-hoc-teamwork problem
statement (Stone et al., AAAI 2010) defines the teammate as an agent
the ego did not co-design and cannot see inside; a deployed robot's
knowledge of a co-worker robot is bounded by its sensors and whatever
the communication channel carries (ADR-003), never by access to the
co-worker's controller internals. The contract encodes exactly that
boundary and nothing stricter.

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | No formal contract — rely on convention and review | [status quo before Phase 0 enforcement] | Zero code | Silent joint-training bugs invalidate every AHT claim; unenforceable at PR review scale |
| B | Full opacity — the partner is invisible except through task effects (no pose observation) | [strictest reading of "black box"] | Strongest-looking ad-hoc claim | Unrealistic: real robots see other robots; forces the ego to infer what any camera/LIDAR provides for free; contradicts the AHT precedents (Liu 2024 RSS, COHERENT, Huriot & Sibai 2025 all let the ego observe partner state) |
| C | Policy-level opacity with behavioural visibility, enforced structurally in code | [ADR-009 §Decision + 2026-05-21 amendment; Stone et al. 2010; Melting Pot's held-out-population design] | Matches deployment reality; machine-enforced, not convention-enforced; keeps the AHT claim honest | Pose visibility slightly weakens the ad-hoc claim (partner-specific adaptation is possible in principle) |

## Decision

**Option C.** The partner is **black-box at the policy level**: the ego
— and every CHAMBER task, baseline, and CONCERTO component — never
accesses a partner's weights, gradients, reward function, or training
data. **No CHAMBER task, baseline, or CONCERTO component may consume
partner-policy information, ever.**

Enforcement is structural, at two points:

1. `chamber.partners.interface._FORBIDDEN_ATTRS` — the attribute shield
   on `PartnerBase` raises `AttributeError` on any policy-access lookup
   (`train`, `learn`, `update`, `update_params`, `fit`, `set_weights`,
   `load_state_dict`, `named_parameters`), so even a caller holding the
   raw partner object cannot reach its policy surface.
2. `chamber.benchmarks.ego_ppo_trainer._assert_partner_is_frozen` — the
   trainer-side gate refuses, at construction time, any partner
   exposing a parameter with `requires_grad=True`, covering adapters
   that bypass `PartnerBase`.

**Pose visibility is not policy access** (per the ADR-009 2026-05-21
amendment, which this ADR elevates from amendment to standing
constraint): a real robot's cameras, LIDAR, and proximity sensors see
other robots' poses, so observing the partner's joint state in
simulation is the same affordance, not a contract violation. The
boundary is *behaviour observable in the world* (allowed) versus
*policy internals* (forbidden).

## Rationale

The contract is the project's realism constraint on partner knowledge:
anything the ego consumes about the partner must be obtainable by a
deployed robot's sensors or the ADR-003 channel. Option B fails realism
in the opposite direction from Option A — it starves the ego of what
deployment provides for free and contradicts every named AHT precedent.
Option C is what the code has enforced since Phase 0; formalizing it
(rather than leaving it as an ADR-009 amendment paragraph) matters now
because ADR-027's admission protocol and baseline set (B-BLIND vs
coupling-aware egos) quantify exactly the value of the behavioural
information this contract permits, and that measurement is meaningless
if the policy boundary can leak.

## Evidence basis (links to reading notes)

- [notes/tier1/construct_validity_in_cooperative_evaluation.md] —
  source 3 (Stone et al. AAAI 2010): the AHT teammate is defined as
  non-co-designed and non-transparent; source 4 (Leibo et al. ICML
  2021): held-out background populations enter the evaluation as
  behavioural black boxes, observable in the world, opaque as policies.
- `chamber.partners.interface` / `chamber.benchmarks.ego_ppo_trainer`
  — the enforcement points as shipped and tested (Phase-0 onward).
- [ADR-009 §Revision history 2026-05-21] — the pose-visibility
  clarification this ADR elevates.

## Consequences

- **Standing prohibition.** Any future task, baseline, wrapper, or
  analysis that reads partner weights, gradients, reward, or training
  data is rejected at review; there is no experimental exemption — a
  design that needs policy access is out of scope for CHAMBER, not a
  config flag away.
- **B-JOINT boundary (ADR-011 as amended).** The jointly-trained
  MAPPO pair baseline trains *as a pair* outside the AHT setting and is
  reported as an upper anchor, never as an ad-hoc method; its training
  procedure is exempt (it is not an ego consuming a partner — it is the
  pair), but its *evaluation* against zoo partners obeys the contract.
- **Partner-set hashing (ADR-028).** Result bundles pin per-partner
  hashes; the hash is computed over the partner's serialized artefact,
  which is custody, not policy access.
- **Other ADRs.** Depends on ADR-006 (the bounds the safety stack may
  assume about partner behaviour are exactly behavioural bounds);
  makes concrete one half of ADR-026's realism story (the other half —
  what the *task* must demand — is the coupling-validity criterion);
  binds ADR-009/ADR-010 partner construction.

## Risks and mitigations

- [risk] Partner-specific adaptation through pose observation weakens
  the ad-hoc claim. → Already documented in ADR-009; the conformal-CBF
  λ re-init on partner swap (ADR-004 §Risks #2) is the safeguard;
  ADR-027's per-partner breakdown reporting makes any overfitting to a
  single partner visible.
- [risk] A future dependency exposes policy internals through a path
  not covered by `_FORBIDDEN_ATTRS` (e.g. a new serialization API). →
  The trainer-side gate is the second net; extending the shield list is
  a test-covered one-line change and does not need a new ADR.

## Reversibility

**Type-1 for the prohibition, Type-2 for the enforcement mechanics.**
Relaxing the policy boundary would retroactively invalidate every AHT
claim published under it — that is the one-way door, and the exit ramp
is a superseding ADR that would have to re-frame the project away from
ad-hoc teamwork. The specific attribute list and gate implementation
are freely evolvable.

## Validation criteria

The contract is validated continuously: the Phase-0 test surface pins
both enforcement points (`AttributeError` from the shield;
`ValueError` from the trainer gate), and every training run's partner
passes `_assert_partner_is_frozen` before tensor allocation. The ADR is
considered Validated once the ADR-027 admission protocol has run A3
(partner-relevance) end-to-end on one admitted task with the contract
in force — demonstrating the boundary permits enough behavioural signal
for coupling-aware egos to outperform partner-blind ones.

## Open questions deferred to a later ADR

- Whether the ADR-003 channel may carry partner-*declared* capability
  metadata (a vendor-style datasheet) without breaching the contract —
  relevant to the Stage-2 CM axis design; declared metadata is not
  policy access but is also not sensor-observable.
