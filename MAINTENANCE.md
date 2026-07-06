# Maintenance

This document states who maintains CONCERTO/CHAMBER, on what cadence,
and under what discipline things are allowed to change. It exists so
that a benchmark consumer can judge whether a number they cite today
will still be interpretable in two years.

## Maintainer and cadence

The project has a **single maintainer**: Farhad Safaei
(<cooperative.physical.ai@gmail.com>). Issues and pull requests are
triaged weekly; leaderboard-submission PRs are acknowledged within
two weeks. Security reports follow [`SECURITY.md`](SECURITY.md).
There is no commercial support and no service-level guarantee; the
verification tooling (`chamber-eval verify`) is deliberately designed
so that results remain checkable without any action from the
maintainer.

## What requires an ADR

Schema and protocol changes are decision-bearing, never incidental.
A new Architecture Decision Record (see
[`adr/ADR-INDEX.md`](adr/ADR-INDEX.md)) is required **before
implementation** for any change to:

- the result-bundle schema
  (`chamber.evaluation.results.SCHEMA_VERSION`, ADR-028) or the
  preregistration schema;
- the communication wire format (`chamber.comm.SCHEMA_VERSION`,
  ADR-003);
- the evaluation protocol — tier ladder, admission checks, reporting
  rules, checkpoint-selection rule (ADR-027);
- a task's physics, success predicate, or stress instrument, or the
  composition of a partner set (ADR-027 §Versioning — this is also a
  version bump, below).

Accepted ADRs are never edited to mean something new; a change of
mind is a new ADR that supersedes the old one.

## Versioned, never mutated

Tasks and partner sets are versioned as `name@vN`. A version, once
results have been published against it, is **immutable**: any change
creates `@v(N+1)`, old results stay interpretable under their
version, and the suite composition is pinned by the generated
manifest (`chamber-eval manifest`). The same applies to evidence:
admission reports, preregistration tags, and committed result
bundles are never edited or re-run in place — a verdict can only be
superseded by a new report under a new task version.

## The open/closed boundary

The partner zoo has a deterministic **70% public / 30% private
split** (ADR-009 as amended). Public members ship their construction
parameters and checkpoints; private members ship only their identity
hashes (SHA-256 over the serialized artifact) and behavioural
fingerprints — their parameters derive from a maintainer-held seed
that is never committed. The private split exists so that a
submitted method can be spot-checked against partners it cannot have
overfitted; the published hashes make that spot-check verifiable
after the fact. Everything else — tasks, protocol, tooling,
evaluation episodes, leaderboard bundles — is open.

## Deprecation discipline

When a task version, partner-set version, or hosted data artifact is
superseded, the superseded version remains available and verifiable
for **at least 24 months** from the deprecation announcement.
Deprecations are announced in [`CHANGELOG.md`](CHANGELOG.md) and on
the affected dataset cards, with the replacement named. Nothing is
deleted inside the window; after it, archives may move to
cold storage (Zenodo deposits are permanent regardless).

## Contact

- Bugs, questions, submissions:
  [GitHub issues](https://github.com/fsafaei/concerto/issues)
- Maintainer: <cooperative.physical.ai@gmail.com>
- Security: see [`SECURITY.md`](SECURITY.md)
