# CONCERTO

[![CI](https://github.com/fsafaei/concerto/actions/workflows/ci.yml/badge.svg)](https://github.com/fsafaei/concerto/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-blue)](https://fsafaei.github.io/concerto/)
[![PyPI](https://img.shields.io/pypi/v/concerto)](https://pypi.org/project/concerto/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/fsafaei/concerto/badge)](https://scorecard.dev/viewer/?uri=github.com/fsafaei/concerto)

**CONCERTO** is the safety stack and ego-AHT training algorithm for
**heterogeneous multi-robot ad-hoc teamwork** (HetMR-AHT).

**CHAMBER** is the matching benchmark suite — ManiSkill v3 wrappers,
partner zoo, evaluation harness, and leaderboard.

The canonical sentence: **"We evaluate CONCERTO on CHAMBER."**

## Why this project?

Real-world multi-robot deployment pairs robots that were not trained together,
have different embodiments and action frequencies, and cannot share policy
weights. Existing benchmarks assume homogeneous teams or full policy access.
CHAMBER provides four heterogeneity axes — action-space (AS),
observation-modality (OM), control-rate (CR), and communication degradation
(CM) — with a pre-registered, ≥20pp comparison protocol and a composite
Heterogeneity Robustness Score (HRS).

CONCERTO addresses all four axes through an exponential CBF-QP safety filter
with a conformal overlay and an OSCBF inner layer, trained with a
frozen-partner ego-AHT variant of HAPPO.

## Install

```bash
git clone https://github.com/fsafaei/concerto.git
cd concerto
pip install uv       # if not already installed
uv sync --group dev
```

## Quick start

```bash
# Print version
uv run concerto

# Run Stage-0 smoke test (M1+ only)
uv run pytest -m smoke -x -v
```

## Documentation

Full documentation at <https://fsafaei.github.io/concerto/>:

- **[Tutorials](https://fsafaei.github.io/concerto/tutorials/hello-spike/)** — step-by-step walkthroughs
- **[How-tos](https://fsafaei.github.io/concerto/how-to/add-partner/)** — goal-oriented recipes
- **[API reference](https://fsafaei.github.io/concerto/reference/api/)** — generated from docstrings
- **[ADR index](https://fsafaei.github.io/concerto/reference/adrs/)** — design decisions
- **[Explanation](https://fsafaei.github.io/concerto/explanation/why-aht/)** — why these choices

## Repository structure

```
src/
├── concerto/     # the METHOD — safety stack + ego-AHT training
└── chamber/      # the BENCHMARK — env wrappers, partner zoo, evaluation
adr/              # Architecture Decision Records (snapshot)
docs/             # Diátaxis documentation
tests/            # unit / property / integration / smoke
spikes/           # pre-registration YAMLs + result archives
scripts/          # local verification and reproduction scripts
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Every PR maps to an ADR or an
issue. DCO (`Signed-off-by`) required for internal authors; CLA-assistant
for external contributors.

## Citation

If you use CONCERTO or CHAMBER in your research:

```bibtex
@software{safaei2026concerto,
  author  = {Safaei, Farhad},
  title   = {{CONCERTO}: Contact-rich Cooperation with Novel Cooperators
             under Embodiment Heterogeneity, Trust bounds, and Opacity},
  year    = {2026},
  url     = {https://github.com/fsafaei/concerto},
  license = {Apache-2.0},
}
```

## Licence

Apache 2.0 — see [LICENSE](LICENSE).
