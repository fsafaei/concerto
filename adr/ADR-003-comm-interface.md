# ADR-003: Communication interface — fixed-format, learned, or both

**Status.** Accepted (2026-05-13) (dependency update 2026-05-08 in light of ADR-007 rev 3 staged rollout)
**Authors.** Farhad Safaei
**Reviewers.** _to fill_
**Tags.** v0.2 §13, §3.9, HMARL

## Context
v0.2 §13 names this as open. The benchmark must support both fixed-format protocols (so any black-box partner can be plugged in) and learned messages (so methods that train them have somewhere to put them). The black-box AHT constraint is the binding design pressure: the partner policy arrives frozen and opaque at deployment, making any protocol that requires joint negotiation of message representations incompatible with the primary setting. At the same time, B6 baselines (learned-comm methods such as HetGPPO and CommFormer) must run on the same benchmark tasks as B5 baselines (no explicit comm), or the comparison is not controlled. The interface design must therefore accommodate both research lines without collapsing into a single mandatory choice.

The fixed-format channel's semantics derive from deterministic-networking standards. IEEE 802.1AS establishes the generalised precision time protocol for distributed clock synchronisation; IEEE 802.1Qbv defines time-aware scheduled traffic with guaranteed latency bounds; IEEE 802.1CB defines frame replication and elimination for redundancy. The 5G-TSN integration model in 3GPP TS 23.501 §5.27 specifies how the 5G system is exposed as a virtual TSN bridge with DS-TT and NW-TT translator functions. CHAMBER's URLLC degradation profiles (`chamber.comm.URLLC_3GPP_R17`) are anchored to 3GPP Release 17 URLLC service-class targets and the 5G-ACIA industrial integration recommendations. The clause-level summaries and the standard-to-measurable-variable mapping live in [`docs/reference/standards.md` §Deterministic networking and 5G-TSN](../docs/reference/standards.md#2-deterministic-networking-and-5g-tsn).

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Fixed-format only | [external; standards bodies] | Simple, partner-agnostic | Throws away learned-comm research |
| B | Learned messages only | [Tier-1 #23 HetGPPO, Tier-3 #27 CommFormer] | Best-in-class for joint-trained settings | Partner must agree on protocol; breaks black-box |
| C | Both — fixed-format wire protocol, learned-message overlay | [Tier-1 #23 HetGPPO] | Supports both research lines; ad-hoc partners use only fixed | Implementation complexity |

## Decision
Adopt **Option C — fixed-format wire protocol as the mandatory baseline, with an opt-in learned-message overlay** available exclusively for jointly-trained B6 baselines. The fixed-format channel is the only channel that black-box ad-hoc partners are required to consume or produce.

## Rationale
HetGPPO's DTDE GNN (`notes/tier1/23_hetgppo.md`, §5) already embodies the hybrid design implicitly: its edge features are fixed-format relative-pose tensors (position + velocity, Euclidean), while the aggregation function and message weights are learned. The black-box partner assumption (`notes/tier1/23_hetgppo.md`, §1 — "full joint training throughout" is HetGPPO's requirement, which B0 cannot satisfy at deployment) means that any learned message representation agreed upon during joint training becomes unavailable to a frozen partner; therefore Option B is structurally incompatible with the primary AHT setting. CommFormer (`notes/tier3/refs.bib #27 CommFormer`) learns a dynamic communication graph, which also requires joint training and cannot be applied to a black-box partner. Restricting to Option A discards the learned-comm research line and prevents direct comparison with B6 baselines; Option C preserves it by making learned messages an opt-in overlay that B5/ad-hoc baselines ignore. The mandatory fixed-format channel — pose broadcast, task-state predicate, and AoI timestamp — is sufficient for the CBF safety layer (v0.2 §6), which intercepts actions at the command level and requires only partner trajectory observations, not partner intent messages.

## Evidence basis (links to reading notes)
- [notes/tier1/23_hetgppo.md] — DTDE GNN architecture (§5, Fig. 2): fixed-format relative-pose edge features as the mandatory observation channel; full joint training as the prerequisite for learned message aggregation weights, establishing why learned-only breaks black-box AHT; behavioral-typing brittleness implies fixed-format pose observations must be noise-robust
- [notes/tier3/refs.bib #27 CommFormer] — dynamic communication graph that learns to construct and exploit topology for coordination; joint-training requirement confirmed in abstract; establishes CommFormer as an opt-in learned-overlay candidate, not a wire-protocol baseline

### Standards (deterministic networking and 5G-TSN)
- [docs/reference/standards.md#ieee-8021as](../docs/reference/standards.md#ieee-8021as) — IEEE 802.1AS, the generalised precision time protocol that gives the fixed-format channel its per-tick clock alignment.
- [docs/reference/standards.md#ieee-8021qbv](../docs/reference/standards.md#ieee-8021qbv) — IEEE 802.1Qbv, time-aware scheduled traffic; anchors the microsecond-grade jitter bounds the URLLC profiles report.
- [docs/reference/standards.md#ieee-8021cb](../docs/reference/standards.md#ieee-8021cb) — IEEE 802.1CB, frame replication and elimination for reliability; the reference for the redundancy variant of the comm-degradation wrapper.
- [docs/reference/standards.md#3gpp-ts-23501-527](../docs/reference/standards.md#3gpp-ts-23501-527) — 3GPP TS 23.501 §5.27, the integration model exposing the 5G system as a virtual TSN bridge with DS-TT / NW-TT translator functions.
- [docs/reference/standards.md#5g-acia-5g-tsn-integration-white-paper](../docs/reference/standards.md#5g-acia-5g-tsn-integration-white-paper) — 5G-ACIA 5G-TSN integration white paper; the industry-side cross-check on factory-floor parameterisation of the URLLC degradation profiles.

## Consequences
- **Project scope**: the benchmark API exposes two observation-bus channels — a mandatory fixed-format channel (pose, task-state predicates, AoI timestamp) and an optional learned-message channel; any partner that does not implement the learned channel is fully interoperable; jointly-trained B6 baselines can use both channels.
- **v0.2 plan sections affected**: §3.9 (communication topology — fixed-format basis for decentralised coordination is confirmed), §13 (interface decision resolved), §6 / §6.2 (CBF safety filter reads from the fixed-format channel only — no learned-message dependency in the safety layer).
- **Other ADRs**: informs ADR-002 (the HARL/HAPPO B5 baseline uses no explicit learned comm — fixed-format channel suffices); informs ADR-004 (CBF safety filter reads partner pose and AoI from the fixed-format channel); informs ADR-009 (partner-zoo policies need only implement the fixed-format channel to be admissible). **Precondition for ADR-007 rev 3 Stage 2 CM spike**: the fixed-format channel (pose, task-state predicate, AoI timestamp) must be scaffolded with a degradation wrapper supporting latency / jitter / drop injection before Stage 2 begins; the spike anchors its sweep range to URLLC + 3GPP Release 17 + arXiv 2501.12792 numbers (latency 1–100 ms, jitter μs–10 ms, drop 10⁻⁶ to 10⁻²) — the channel scaffold must accept these ranges without saturating the QP solver in the inner CBF loop.
- **Compute / time / hiring**: two-channel API adds benchmark engineering overhead (estimated 1 week); the AoI timestamp field in the fixed-format channel requires a hardware or simulation clock synchronisation mechanism; the comm-degradation wrapper for ADR-007 Stage 2 adds approximately 0.5 person-week and should be sequenced before Stage 1 completes so it is available when Stage 2 starts.

## Risks and mitigations
- [**Learned-message overlay violates black-box assumption if accessed at inference time**] A B6 baseline that reads a partner's learned message embeddings requires instrumenting the partner's internal communication output, breaking the black-box contract. Mitigation: enforce at the benchmark API level that the learned-message channel is populated only by agents that are flagged as jointly-trainable; ad-hoc and black-box partner instances expose a null/empty learned channel and any baseline consuming it must handle the null case gracefully, with a mandatory fixed-format-only ablation reported alongside every B6 result.
- [**Fixed-format channel bandwidth may be insufficient for manipulation coordination**] Pose and task-state predicates alone may not convey enough information for tight bimanual manipulation coordination (e.g., which object face is being grasped, current contact-force estimate). Mitigation: extend the fixed-format schema with a small typed-predicate extension field (e.g., grasp-side, object-class, contact-force scalar) defined at benchmark initialisation; the field is optional and defaults to zeros, preserving partner-agnostic interoperability while allowing richer coordination when partners support it.

## Reversibility
**Type-2** (two-way door). The wire protocol is a benchmark API contract, not a trained artifact; switching channel schema or disabling the learned-message overlay costs an API version bump and partner-adapter updates, estimated at 1–2 person-weeks. The practical reversal window is before Phase-1 B6 baseline runs accumulate learned-channel dependencies in published checkpoints.

## Validation criteria
By Phase-1 end: at least one B6 baseline (HetGPPO or CommFormer AHT fork) using the learned-message overlay and at least one B5 baseline (HAPPO, fixed-format only) run on the same benchmark task and report task-success rate, with the B6 baseline also reporting a fixed-format-only ablation to isolate learned-comm contribution.

## Open questions deferred to a later ADR
- (#design-decision) What typed predicates beyond pose and velocity should the fixed-format extension field carry for manipulation tasks (grasp-side, contact-force scalar, object class)? Defer to task-design sprint before Phase-1 benchmark lock.
- (#verify, from `notes/tier1/23_hetgppo.md`) Can the DTDE GNN's communication be cleanly split into a fixed-format edge-feature path and a learned aggregation path such that freezing the learned path still yields a usable fixed-format-only baseline? Confirm before committing HetGPPO as B3/B6.
- (#verify) Does CommFormer's dynamic graph construction degrade gracefully when the learned-message channel is null (black-box partner setting), or does it require all agents to have an active learned channel? Confirm when CommFormer note #27 is written.
- (#design-decision, from ADR-007 rev 3) Should the comm-degradation wrapper expose AoI directly (note 41 Ballotta & Talak) as the latency proxy for ADR-007's Stage 2 CM spike, or end-to-end round-trip time? AoI is more theoretically grounded and matches the ADR-008 HRS bundle latency-axis instrumentation, but RTT is cheaper to log; the choice affects whether ADR-008 Option A (latency × drop × partner-familiarity) is directly comparable across baselines.

## Revision history
- 2026-05-13 revision: anchors the fixed-format channel in the IEEE 802.1 TSN family (802.1AS / 802.1Qbv / 802.1CB) and the 3GPP TS 23.501 §5.27 5G-TSN integration model in Context and Evidence basis. No change to the Decision. Ties to the public standards reference at docs/reference/standards.md.
- 2026-05-13 status re-classification: status changed from Proposed to **Provisional** under the new ADR status taxonomy (see [ADR-INDEX §Status taxonomy](ADR-INDEX.md#status-taxonomy)); no Decision content is altered.
- 2026-05-13 lock: promoted to **Accepted (2026-05-13)** under the solo-developer working policy in ADR-INDEX (M2 comm stack is merged on `main` and the fixed-format channel is in active use); no Decision content is altered.
