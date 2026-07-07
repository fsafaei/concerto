---
viewer: false
license: apache-2.0
pretty_name: CHAMBER-Bench v1.0 reference trajectories
tags:
  - robotics
  - reinforcement-learning
  - multi-agent
  - benchmark
---

# chamber-bench-reference-trajectories

The reference and probe evidence for **CHAMBER-Bench v1.0**, released
with the [fsafaei/concerto](https://github.com/fsafaei/concerto)
repository: the episodes that pin down *who the partners are* and the
exploratory runs that sized the campaigns.

## What is in this dataset

- `partner-fingerprints/` — for every partner-set member, a
  verifiable probe bundle (the preregistered probe episodes it was
  fingerprinted on) plus a per-set `fingerprints.json` summary; also
  the `cocarry_partners-v2-dropped-candidates/` record documenting
  the four learned candidates rejected by the committed competence
  floor. A *fingerprint* is a deterministic behavioural summary of a
  partner measured on fixed probe episodes — it identifies a partner
  without revealing its parameters.
- `power-pilots/` — the power-pilot bundles: the `run_purpose: power`
  exploratory runs that sized the campaign sample budgets before
  preregistration. They are committed evidence, deliberately excluded
  from every leaderboard manifest — pilots inform design, they never
  contribute to a reported number.
- `manifest.json`, `SHA256SUMS.txt` — file list and digests; check a
  download with `sha256sum -c SHA256SUMS.txt`.

## How to verify

Every probe bundle and power-pilot bundle is an ADR-028 result
bundle; from a repository checkout:

```bash
uv run chamber-eval verify <bundle-directory>
```

The evaluation protocol these bundles feed, with worked examples, is
documented in `docs/explanation/evaluation-protocol.md`. The
admission-report archives that decide which tasks count are released
alongside the rows they support, in
`chamber-bench-leaderboard-bundles`.

## Provenance

Generated in simulation by the repository's tooling under tag-locked
preregistrations, July 2026. Probe episodes for private partner
members were produced with the maintainer-held seed; the episodes
reveal behaviour, not parameters. No personal data of any kind.

## Intended use

Auditing partner identity and competence floors; regression-testing
partner behaviour against committed fingerprints; auditing how the
campaign sample sizes were chosen. Training on probe episodes and
then submitting to the leaderboard defeats the unseen-partner
condition and is out of scope.

## Limitations

Probe coverage follows the preregistered probe grid; power-pilot
coverage exists for the runs the v1.0 campaigns actually used to size
their budgets.

## Licence and citation

Apache-2.0, same as the repository. Cite via the repository's
`CITATION.cff` (DOI `10.5281/zenodo.20128468`). Maintainer contact
and deprecation policy: `MAINTENANCE.md` in the repository.
