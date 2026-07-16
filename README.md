<p align="center">
  <em>Contact-rich coordination with opaque, heterogeneous teammates.</em><br/>
  <strong>CONCERTO</strong> is the method.
  <strong>CHAMBER</strong> is the benchmark.
  We evaluate CONCERTO on CHAMBER.
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20128468">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20128468.svg"/></a>
  <a href="https://github.com/fsafaei/concerto/actions/workflows/ci.yml">
    <img alt="CI"   src="https://github.com/fsafaei/concerto/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://fsafaei.github.io/concerto/">
    <img alt="Docs" src="https://img.shields.io/badge/docs-latest-blue"/></a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"/></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/fsafaei/concerto">
    <img alt="OpenSSF Scorecard" src="https://api.scorecard.dev/projects/github.com/fsafaei/concerto/badge"/></a>
  <a href="https://pypi.org/project/concerto-multirobot/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/concerto-multirobot"/></a>
</p>

## What this is

CHAMBER is a simulation benchmark for **physically-coupled ad-hoc
cooperation**: two robot arms jointly manipulate one object (carry a
rigid bar; hand a part over and place it) while the evaluated robot —
the *ego* — works with a **black-box partner**, meaning a teammate
whose policy, parameters, and training history it cannot see and that
it was never trained with. CONCERTO is the companion method: a
safety-filter stack and a training loop for exactly this setting.
This repository ships both, plus everything needed to reproduce and
verify every number below.

**The claim this benchmark supports.** A leaderboard row supports one
kind of claim: *this method, controlling one arm, achieves the stated
joint-success rate with previously-unseen black-box partners on this
physically-coupled task, with the stated confidence interval.* Every
row is recomputed at render time from a committed result bundle that
anyone can verify with one command. Nothing broader is claimed — no
cross-task ranking, no scalar composite, and no
heterogeneity-robustness aggregate in v1.0.

**Under what assumptions.** Both scored tasks passed a preregistered
admission protocol showing that a reference pair solves them, one
robot alone cannot, and destroying the physical coupling destroys
performance ([ADR-027](adr/ADR-027-chamber-bench-v1-protocol.md)).
Partners are frozen during evaluation and drawn from a versioned
partner zoo with a public/private split, so a method cannot overfit
the full evaluation set. CPU evaluation is byte-identical under the
pinned `uv.lock` and a root seed, which is what makes committed
bundles recomputable.

**With what limitations.** Everything is simulation; no real-robot
results exist. Teams are exactly two agents on the listed embodiments,
and the v1.0 partner zoo contains no vision-based partners. Measured
heterogeneity effects on co-carry are null so far, so v1.0 measures
ad-hoc-teamwork competence, not heterogeneity robustness — see
[Limitations](#limitations).

## Install

[`uv`](https://docs.astral.sh/uv/) is the package manager; versions
are pinned by the committed `uv.lock`. Do not invoke pip directly.

```bash
git clone https://github.com/fsafaei/concerto.git
cd concerto
pip install uv && uv sync --group dev
```

Python 3.11 or 3.12. The training stack (the HARL fork, published as
the `harl-aht` distribution) resolves automatically as a runtime
dependency; no extra install step is needed.

The package is also published to
[PyPI](https://pypi.org/project/concerto-multirobot/) as
`concerto-multirobot` for library use. The quickstart and all
benchmark reproduction need the repository checkout, since the
committed result bundles and preregistration tags live here.

## Five-minute quickstart

Run one evaluation and verify one shipped result. The commands below
are executed verbatim by CI on every push (jobs `smoke-eval` and
`readme-quickstart` in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml)), so they are
guaranteed to work on a clean checkout.

**1. Run a real evaluation on the CPU diagnostic task** (~1 minute).
This produces a *result bundle*: a directory containing every episode
record, the exact command, the git state, and SHA-256 checksums of all
files.

```bash
uv run chamber-eval run --task mpe_cooperative_push --policy random \
  --partner scripted_heuristic --seeds 2 --episodes 5 --out out/quickstart-bundle
uv run chamber-eval verify out/quickstart-bundle
```

Expected: the run reports `10 episodes; success IQM 0.667 [0.000, 1.000]`
(byte-identical on CPU under the pinned lockfile), and `verify` prints
a check table ending in `verify: PASS`. On a locally-modified checkout
add `--allow-dirty` to both commands; such bundles are marked dirty
and are ineligible for the leaderboard.

**2. Verify a shipped leaderboard row** (~10 seconds). This re-checks
every file hash in the committed bundle, recomputes the headline
statistics from the raw per-episode records, and re-derives the
preregistration tag:

```bash
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/ref-script-2026-07-05
```

Every cell of the leaderboard below can be checked the same way, and
[docs/how-to/reproduce-results.md](docs/how-to/reproduce-results.md)
gives the full re-run command for every row.

## The benchmark

The suite is pinned by the `chamber.tasks` registry
([ADR-027](adr/ADR-027-chamber-bench-v1-protocol.md) §Decision):
every task carries exactly one tier and an admission status backed by
committed evidence. `ADMITTED` is earned, never asserted —
`chamber-eval admission` executes the A1–A4 protocol under a
tag-locked preregistration and writes the immutable report the status
cites (see `spikes/results/admission/`). The table is generated; CI
fails if it drifts.

<!-- CHAMBER-BENCH-TASK-TABLE:BEGIN — generated by scripts/render_task_table.py; edit src/chamber/tasks/, not this table -->

| Task | Tier | Status | Axis validity | Task card |
|------|------|--------|---------------|-----------|
| `mpe_cooperative_push@v1` — MPE cooperative push (CPU diagnostic) | 0 (rig diagnostics) | DIAGNOSTIC | all untested | [mpe_cooperative_push](docs/reference/tasks/mpe_cooperative_push.md) |
| `stage0_smoke@v1` — Stage-0 tri-embodiment smoke rig | 0 (rig diagnostics) | DIAGNOSTIC | all untested | [stage0_smoke](docs/reference/tasks/stage0_smoke.md) |
| `stage1_pickplace_as@v1` — Stage-1 pick-and-place — action-space (AS) control | 1 (controls) | CONTROL | AS `invalid` | [stage1_pickplace_as](docs/reference/tasks/stage1_pickplace_as.md) |
| `stage1_pickplace_om@v1` — Stage-1 pick-and-place — observation-modality (OM) control | 1 (controls) | CONTROL | all untested | [stage1_pickplace_om](docs/reference/tasks/stage1_pickplace_om.md) |
| `cocarry@v1` — Co-carry — rigid dual-arm bar transport | 2 (admitted cooperation tasks) | ADMITTED | AS `null` | [cocarry](docs/reference/tasks/cocarry.md) |
| `handover_place@v1` — Handover-and-place under takt pressure | 2 (admitted cooperation tasks) | ADMITTED | all untested | [handover_place](docs/reference/tasks/handover_place.md) |
| `amr_handover_dynamic@v0` — AMR dynamic handover (spec-only candidate) | 3 (documented candidates and closures) | CANDIDATE | all untested | [amr_handover_dynamic](docs/reference/tasks/amr_handover_dynamic.md) |
| `co_hold_secure@v0` — Co-hold-and-secure (spec-only candidate) | 3 (documented candidates and closures) | CANDIDATE | all untested | [co_hold_secure](docs/reference/tasks/co_hold_secure.md) |
| `coinsert@v1` — Co-insert — hold-and-insert (closed; open challenge) | 3 (documented candidates and closures) | CLOSED | all untested | [coinsert](docs/reference/tasks/coinsert.md) |

<!-- CHAMBER-BENCH-TASK-TABLE:END -->

## Leaderboard — CHAMBER-Bench v1.0

Per-task tables only; no scalar composite
([ADR-027](adr/ADR-027-chamber-bench-v1-protocol.md) §Decision,
reporting rules). Every row is recomputed from committed result
bundles that pass the full `chamber-eval verify` check table at render
time — an unverifiable entry refuses the whole render — under the
tag-locked campaign preregistration. Two rows carry honest labels and
are not competitors: REF-SCRIPT is the *oracle reference* (a scripted
controller with privileged access, shown as the solvability ceiling),
and B-JOINT is the *non-AHT upper anchor* (a jointly-trained pair
evaluated with its own training partner, which breaks the unseen-partner
condition by construction). The block is generated by
`scripts/render_leaderboard_table.py`; CI fails if it drifts.

<!-- CHAMBER-BENCH-LEADERBOARD:BEGIN — generated by scripts/render_leaderboard_table.py; edit spikes/results/benchmark/, not this table -->
#### cocarry-v1

| Row | Label | Success IQM [95% CI] | Success mean | Stress p90 (N) | Per-partner range | Seeds | Bundles |
|---|---|---|---|---|---|---|---|
| REF-SCRIPT | *oracle reference* | 1.000 [1.000, 1.000] | 1.000 | 102.0 | 1.00 (imp_stiff_low) … 1.00 (imp_stiff_low) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/ref-script-2026-07-05` |
| B-RND |  | 0.000 [0.000, 0.000] | 0.001 | 277.8 | 0.00 (imp_stiff_low) … 0.00 (imp_blend_b) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/b-rnd-2026-07-05` |
| B-STAT |  | 0.492 [0.396, 0.591] | 0.496 | 101.4 | 0.16 (imp_lag_bounded) … 0.66 (imp_blend_b) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/b-stat-2026-07-05` |
| B-BLIND |  | 1.000 [1.000, 1.000] | 0.967 | 107.6 | 0.86 (imp_lag_bounded) … 1.00 (imp_stiff_low) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/b-blind-2026-07-06` |
| B-AHT |  | 1.000 [1.000, 1.000] | 0.961 | 108.4 | 0.88 (imp_lag_bounded) … 1.00 (imp_stiff_low) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/b-aht-2026-07-06` |
| B-JOINT | *non-AHT upper anchor* | 1.000 [1.000, 1.000] | 0.944 | 41.2 | 0.94 (frozen_cocarry_joint) … 0.94 (frozen_cocarry_joint) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/cocarry-v1/b-joint-seed0-2026-07-06`<br>`spikes/results/benchmark/cocarry-v1/b-joint-seed1-2026-07-06`<br>`spikes/results/benchmark/cocarry-v1/b-joint-seed2-2026-07-06`<br>`spikes/results/benchmark/cocarry-v1/b-joint-seed3-2026-07-06`<br>`spikes/results/benchmark/cocarry-v1/b-joint-seed4-2026-07-06` |

*Solvability anchor: matched scripted reference 1.000 (admission A1, `spikes/results/admission/cocarry-2026-07-05/`).*
*Success IQM is the preregistered `iqm_success_rate`: the interquartile mean over per-episode success indicators; it is intentionally conservative away from saturation (e.g., handover REF-SCRIPT mean 0.338 vs IQM 0.176) and trims minority cells at saturation (e.g., B-AHT mean 0.961 vs IQM 1.000 — see the per-partner range).*

#### handover-v1

| Row | Label | Success IQM [95% CI] | Success mean | Stress p90 (N) | Per-partner range | Seeds | Bundles |
|---|---|---|---|---|---|---|---|
| REF-SCRIPT | *oracle reference* | 0.176 [0.072, 0.284] | 0.338 | 211.5 | 0.13 (presenter_mismatch_45) … 0.55 (presenter_mismatch_30) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/handover-v1/ref-script-2026-07-06` |
| B-RND |  | 0.000 [0.000, 0.000] | 0.000 | 3858.1 | 0.00 (presenter_mismatch_30) … 0.00 (presenter_mismatch_30) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/handover-v1/b-rnd-2026-07-06` |
| B-STAT |  | 0.048 [0.000, 0.140] | 0.274 | 225.6 | 0.05 (presenter_mismatch_45) … 0.50 (presenter_mismatch_30) | 0, 1, 2, 3, 4 | `spikes/results/benchmark/handover-v1/b-stat-2026-07-06` |

*Solvability anchor: matched presenter 1.000 (Gate-0 Limb 1, `spikes/results/handover-place-gate0-2026-06-26/`). Leaderboard cells are the measured coupling-valid (mismatch) region by design — the low oracle number is the mismatch penalty, not unsolvability.*
*Success IQM is the preregistered `iqm_success_rate`: the interquartile mean over per-episode success indicators; it is intentionally conservative away from saturation (e.g., handover REF-SCRIPT mean 0.338 vs IQM 0.176) and trims minority cells at saturation (e.g., B-AHT mean 0.961 vs IQM 1.000 — see the per-partner range).*
<!-- CHAMBER-BENCH-LEADERBOARD:END -->

Row ids follow the preregistered baseline set: B-RND (random
actions), B-STAT (static, hold-position), B-BLIND (learned ego with
partner observations masked), B-AHT (learned ad-hoc-teamwork ego).
Reproduction command for every row:
[docs/how-to/reproduce-results.md](docs/how-to/reproduce-results.md).

## Validity

A score on this benchmark means something only if the task actually
forces cooperation, so every claim rests on the coupling-validity
criterion of
[ADR-026](adr/ADR-026-coupling-validity-criterion.md). Tier-1 control
tasks are kept precisely because they are solvable by one robot alone:
they separate plain manipulation skill from cooperation, and a method
that gains on a control is measuring something other than teamwork.
Admission ([ADR-027](adr/ADR-027-chamber-bench-v1-protocol.md)) is
what a task must earn before its scores count, by three preregistered
checks: a reference pair solves it (A1), the best single robot cannot
(A2), and a scripted ego stripped of exactly its coupling channel
collapses (A3). A fourth clause (A4) gates the *measuring instrument*
rather than the task: any policy used as the fixed reference for a
cross-partner contrast must itself succeed across the admitted
partner set, or the contrast is inadmissible.
Co-carry and handover-and-place passed all three;
pick-and-place passed A1 but failed A2 and A3 and is retained as a
Tier-1 control — the protocol demonstrably rejects tasks. The
heterogeneity-robustness score (HRS, ADR-008), the project's original
scalar aggregate, is suspended in v1.0 because no task×axis cell has
a validated heterogeneity effect yet, and averaging nulls into a
headline number would manufacture a ranking out of nothing. Until a
validated cell exists, per-task success tables with confidence
intervals are the only reported currency. The full protocol, with the
co-carry admission as a worked example, is at
[docs/explanation/evaluation-protocol.md](docs/explanation/evaluation-protocol.md).

## Submitting a leaderboard entry

Fork, run the preregistered cells against the public partner set,
open a PR containing your result bundle; CI re-runs
`chamber-eval verify` on it, and the maintainer may spot-check your
method against the private partner split before merge. Full protocol:
[docs/how-to/submit-leaderboard.md](docs/how-to/submit-leaderboard.md).

## Limitations

- **Simulation only.** All results are ManiSkill v3 / SAPIEN (plus
  one CPU diagnostic task); no real-robot results exist.
- **Two agents.** Every task is one ego and one partner; larger teams
  are out of scope for v1.0.
- **Listed embodiments only.** Results cover the embodiments named in
  the task cards (Panda-class arms on the scored rows); an xArm6
  variant was studied and is documented in ADR-026's revision log,
  but no leaderboard row uses it.
- **Heterogeneity effects are null on co-carry so far.** Partner-style
  and embodiment heterogeneity were measured and found null under
  fair matching (ADR-026 §Open questions); v1.0 therefore measures
  ad-hoc-teamwork competence with black-box partners, not
  heterogeneity robustness, and the HRS aggregate stays suspended.
- **No vision-based partners.** The v1.0 partner zoo is scripted
  impedance controllers, scripted presenters, and frozen learned
  policies; vision-language-action partners are deferred stubs.

## Documentation

Full documentation: [**fsafaei.github.io/concerto**](https://fsafaei.github.io/concerto/)

- [Evaluation protocol](docs/explanation/evaluation-protocol.md) — run classes, admission, reporting rules, versioning.
- [Reproduce the results](docs/how-to/reproduce-results.md) — one command block per leaderboard row.
- [Submit a leaderboard entry](docs/how-to/submit-leaderboard.md).
- [Datasheet](docs/reference/datasheet.md) — the released data artifacts, described.
- [Architecture](docs/explanation/architecture.md) — the two packages, the safety stack, repository layout.
- [Positioning](docs/explanation/positioning.md) — why this exists, prior work, the six heterogeneity axes, roadmap.
- [FAQ](docs/explanation/faq.md) · [Glossary](docs/reference/glossary.md) · [ADR index](adr/ADR-INDEX.md)
- [MAINTENANCE.md](MAINTENANCE.md) — who maintains this, versioning and deprecation discipline, contact.

## Contributing

Open from the first commit; PRs welcome. Read
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the development flow,
commit-signing and DCO requirements, and the contributor license
agreement (ADR-012). Every PR cites the ADR section it touches;
protocol changes require a new ADR, never an edit to an accepted one.
Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Security
policy: [`SECURITY.md`](SECURITY.md).

## Citation

If you use CONCERTO or CHAMBER in your research, please cite the
archived software release:

```bibtex
@software{safaei2026concerto,
  author       = {Safaei, Farhad},
  title        = {{CONCERTO} and {CHAMBER}: Contact-rich Coordination
                  with Opaque, Heterogeneous Teammates},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20128468},
  url          = {https://doi.org/10.5281/zenodo.20128468},
  note         = {arXiv preprint forthcoming},
}
```

Citation metadata is also in [`CITATION.cff`](CITATION.cff), so GitHub
renders a "Cite this repository" button.

## License

Apache 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The
safety stack composes
[Wang et al. 2017](https://ieeexplore.ieee.org/document/7989121)
(decentralised exponential control barrier functions),
[Huriot & Sibai 2025](https://arxiv.org/abs/2409.18862) (conformal
control barrier functions), and
[Morton & Pavone 2025](https://arxiv.org/abs/2503.17678)
(operational-space control barrier functions). CHAMBER is a wrapper
layer over [ManiSkill v3](https://github.com/haosulab/ManiSkill) and
depends on a fork of [HARL](https://github.com/PKU-MARL/HARL) for the
training stack.
