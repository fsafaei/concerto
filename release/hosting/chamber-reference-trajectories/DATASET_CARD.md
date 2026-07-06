# chamber-reference-trajectories

The reference and probe evidence for **CHAMBER-Bench v1.0**, released
with the [fsafaei/concerto](https://github.com/fsafaei/concerto)
repository: the episodes that pin down *who the partners are* and
*why the tasks count*.

## What is in this dataset

- `partner-fingerprints/` — for every partner-set member, a
  verifiable probe bundle (the preregistered probe episodes it was
  fingerprinted on) plus a per-set `fingerprints.json` summary; also
  the `cocarry_partners-v2-dropped-candidates/` record documenting
  the four learned candidates rejected by the committed competence
  floor. A *fingerprint* is a deterministic behavioural summary of a
  partner measured on fixed probe episodes — it identifies a partner
  without revealing its parameters.
- `admission/` — the admission-report archive for each admission
  decision: `cocarry-2026-07-05` (ADMITTED),
  `handover_place-2026-07-05` (ADMITTED), and
  `stage1_pickplace_as-2026-07-05` (CONTROL — the protocol's
  falsification example). Each contains the human-readable report,
  the machine-readable `admission_report.json`, and one verifiable
  sub-bundle per check (A1 solvability, A2 single-robot
  infeasibility, A3 partner-relevance).
- `manifest.json`, `SHA256SUMS` — file list and digests; check a
  download with `sha256sum -c SHA256SUMS`.

## How to verify

Every probe bundle and admission sub-bundle is an ADR-028 result
bundle; from a repository checkout:

```bash
uv run chamber-eval verify <bundle-directory>
```

The admission protocol itself, with these archives as the worked
examples, is documented in
`docs/explanation/evaluation-protocol.md`.

## Provenance

Generated in simulation by the repository's tooling under tag-locked
preregistrations, July 2026 (tags
`prereg-admission-cocarry-2026-07-05`,
`prereg-admission-handover-place-2026-07-05`,
`prereg-admission-stage1-pickplace-as-2026-07-05`). Probe episodes
for private partner members were produced with the maintainer-held
seed; the episodes reveal behaviour, not parameters. No personal
data of any kind.

## Intended use

Auditing partner identity and competence floors; auditing admission
verdicts; regression-testing partner behaviour against committed
fingerprints. Training on probe episodes and then submitting to the
leaderboard defeats the unseen-partner condition and is out of
scope.

## Limitations

Probe coverage follows the preregistered probe grid; admission
evidence exists for the three tasks that entered the v1.0 campaign.

## Licence and citation

Apache-2.0, same as the repository. Cite via the repository's
`CITATION.cff` (DOI `10.5281/zenodo.20128469`). Maintainer contact
and deprecation policy: `MAINTENANCE.md` in the repository.
