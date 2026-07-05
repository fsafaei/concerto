# ADR-028: Result-bundle schema v3 and preregistration versioning

**Status.** Accepted (2026-07-05)
**Authors.** Farhad Safaei
**Reviewers.** _solo lock per ADR-INDEX working policy_
**Tags.** v0.2 §3.7; schema lifecycle; depends on ADR-016, ADR-027; invariants I8/I9.

## Context

Two schema debts block the CHAMBER-Bench v1.0 protocol (ADR-027).

First, the result-bundle schema
(`chamber.evaluation.results.SCHEMA_VERSION = 2`, locked by ADR-016)
predates the benchmark framing: a v2 bundle does not carry the task
version, partner-set identity, or reproduction provenance that ADR-027's
versioned, per-task leaderboard requires — today that context lives in
per-archive conventions (`repro_command.txt`, `SHA256SUMS.txt`,
report-file prose) that each spike re-invents.

Second, the handover-and-place Gate-0 pre-registration had to be
written **outside** `chamber.evaluation.prereg.PreregistrationSpec`:
the spec models the legacy *axis-form* (condition tables for the
ADR-007 staged spikes) and cannot express a document-form
pre-registration (decision rules, power simulation, verdict spaces,
frozen thresholds). The Gate-0 YAML shipped as a free-form document
under its own tag discipline, and a docstring in
`chamber.benchmarks.cocarry_freeze` already forward-references a
`PREREG_SCHEMA_VERSION` constant that does not exist. Preregistration
needs the same versioned-schema treatment results got in ADR-016 (I9:
decision first, code second — this ADR is the decision; the code lands
in the implementation follow-up, not with this ADR).

## Considered alternatives

| # | Option | Source / advocate | Pros | Cons |
|---|--------|-------------------|------|------|
| A | Keep v2 + per-archive conventions (sidecar files, report prose) | [status quo] | No schema churn | Provenance is convention, not contract; ADR-027 manifest/leaderboard tooling would parse prose; prereg stays un-modeled |
| B | v3 as a breaking migration — rewrite committed v2 archives in place | [tidiness argument] | One schema everywhere | Violates I8 (immutable archives); rewrites history the tags certify |
| C | Additive v3 + `PREREG_SCHEMA_VERSION = 1` (document-form alongside legacy axis-form); v2 readable forever, never migrated | [ADR-016 lifecycle precedent] | Contract-level provenance; prereg becomes schema-checked; history untouched | Two readable result versions and two prereg forms to maintain |

## Decision

**Option C.** One line: **result bundles gain a v3 schema whose
provenance fields make every bundle independently reproducible and
leaderboard-addressable; pre-registration gains its own versioned
schema with a document-form model; v2 archives stay readable forever
and are never migrated in place.**

This ADR **authorizes** the following; implementation is a separate
code change (I9 kept clean — no `SCHEMA_VERSION` literal changes land
with this decision):

### 1. Result-bundle schema v3

`chamber.evaluation.results.SCHEMA_VERSION` bumps **2 → 3**, adding
top-level fields:

- `git_sha` — the launch commit, captured once per run.
- `package_version` — the installed distribution version.
- `task_id` + `task_version` — the ADR-027 `task_id@vN` identity.
- `partner_set_id` — with **per-partner hashes** (SHA-256 over each
  partner's serialized artefact; custody, not policy access, per
  ADR-018).
- `seed_schedule` — the explicit seed list/derivation, not just a root
  seed.
- `repro_command` — the exact reproduction invocation (today's
  `repro_command.txt` convention, promoted into the bundle).
- Platform fingerprint — OS, Python, key dependency versions, device.
- SHA-256 **file manifest** over the bundle's artefacts (today's
  `SHA256SUMS.txt` convention, promoted into the bundle).

### 2. Preregistration versioning

A `PREREG_SCHEMA_VERSION = 1` constant is introduced in
`chamber.evaluation.prereg` (making the existing
`chamber.benchmarks.cocarry_freeze` docstring reference real), covering
**two coexisting forms**:

- **Legacy axis-form** — the existing `PreregistrationSpec` condition
  tables (ADR-007 spikes); unchanged, still valid.
- **Document-form** — a generalized model for what the
  handover-and-place Gate-0 prereg had to improvise: frozen decision
  rules, verdict spaces, power/pre-check artefacts, threshold tables,
  revision + tag identity. Schema-checked, so "the prereg is frozen"
  becomes machine-verifiable (file blob vs tag blob, as Gate-0 did by
  hand).

### 3. The `chamber-eval` contract

- `chamber-eval run` — executes an evaluation and **emits a v3
  bundle**; refuses to run if the referenced prereg document fails
  schema validation or tag-blob verification.
- `chamber-eval verify` — takes a committed bundle and re-checks its
  file manifest, prereg linkage, and provenance fields; this is the
  command an external reproducer runs first.

### 4. Compatibility rule (binding)

**v2 archives are readable forever and are never migrated in place
(I8).** Readers dispatch on `schema_version`; v3-only tooling
(manifest generation, leaderboard rendering) treats v2 bundles as
historical inputs with explicitly-absent provenance, never as errors.

## Rationale

ADR-027's versioning and reporting rules are only enforceable if the
bundle itself carries task/partner-set identity and reproduction
provenance — otherwise the manifest points at prose. The prereg schema
closes the gap Gate-0 exposed: the project's strongest methodological
asset (tag-frozen pre-registration) currently has schema support only
for the axis-form that ADR-026 retired from the public story. The
additive path follows the ADR-016 precedent, which already established
the lifecycle (bump, dispatch on version, never rewrite archives) and
is the reason option B is rejected rather than debated.

## Evidence basis (links to reading notes)

- [notes/tier1/construct_validity_in_cooperative_evaluation.md] —
  source 6 (Gorsane et al. 2022): pre-committed, documented protocol as
  the reproducibility corrective; the document-form prereg is that
  commitment made schema-checkable.
- [ADR-016] — the schema-lifecycle precedent (v1→v2 bump, typed
  dispatch, immutable archives).
- `spikes/preregistration/handover_place/gate0.yaml` +
  `spikes/results/handover-place-gate0-2026-06-26/` — the improvised
  document-form pattern and the by-hand blob-vs-tag verification this
  ADR systematizes.

## Consequences

- **Implementation scope (follow-up code change, not this ADR):** the
  v3 model + reader dispatch in `chamber.evaluation.results`; the
  prereg constant + document-form model in
  `chamber.evaluation.prereg`; the `chamber-eval run`/`verify`
  surfaces; tests pinning v2 readability.
- **ADR-027:** unblocked — manifest and leaderboard tooling consume v3
  fields.
- **ADR-016:** extended, not superseded; its SpikeRun lifecycle rules
  carry over to v3 unchanged.
- **ADR-014 / ADR-017:** safety-report and observability schemas are
  untouched; `concerto.safety.reporting.SCHEMA_VERSION` (3) and
  `chamber.comm` `SCHEMA_VERSION` (1) are out of scope.

## Risks and mitigations

- [risk] Scope creep: the v3 field list grows during implementation. →
  The field list above is the authorized set; additions need an ADR
  amendment, not a code-review decision.
- [risk] Two prereg forms drift apart in validation strictness. → One
  loader, one version constant, shared tag-blob verification path;
  axis-form is frozen legacy, all new preregs use document-form.
- [risk] Per-partner hashing is read as policy access. → ADR-018
  records the custody/access distinction explicitly.

## Reversibility

**Type-2.** Additive schema evolution with version dispatch is cheap to
extend and cheap to supersede; the only irreversible commitment is the
compatibility rule itself, which is I8 restated and not genuinely
revisitable.

## Validation criteria

The implementation follow-up is accepted when: (1) a v3 bundle
round-trips through `chamber-eval run` → `chamber-eval verify` on a
clean checkout with only the bundle and the repo; (2) every committed
v2 archive still loads through the public reader API, byte-identical
interpretation, proven by tests; (3) the handover-and-place Gate-0
prereg re-expressed in document-form validates against
`PREREG_SCHEMA_VERSION = 1` with its original tag-blob check passing;
(4) `git grep PREREG_SCHEMA_VERSION` finds a defined constant behind
every reference.

## Open questions deferred to a later ADR

- Whether the platform fingerprint should include a GPU-determinism
  attestation (Tier-2 runs are GPU-hosted; byte-identity is a CPU
  guarantee under ADR-002).
- Whether `chamber-eval verify` should re-execute a bundle's
  `repro_command` in a sandboxed smoke mode (full re-run verification)
  or remain integrity-only in v1.0 (current answer: integrity-only).
