<p align="center">
  <a href="https://github.com/fsafaei/concerto">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="docs/assets/dark_mode.png">
      <img alt="CONCERTO" src="docs/assets/light_mode.png" width="420"/>
    </picture>
  </a>
</p>

<p align="center">
  <em>Contact-rich coordination with opaque, heterogeneous teammates &mdash;
  with high-probability safety bounds.</em><br/>
  <strong>CONCERTO</strong> is the method.
  <strong>CHAMBER</strong> is the benchmark.
  We evaluate CONCERTO on CHAMBER.
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.20128469">
    <img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.20128469.svg"/></a>
  <a href="https://github.com/fsafaei/concerto/actions/workflows/ci.yml">
    <img alt="CI"   src="https://github.com/fsafaei/concerto/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://fsafaei.github.io/concerto/">
    <img alt="Docs" src="https://img.shields.io/badge/docs-latest-blue"/></a>
  <a href="https://arxiv.org/abs/XXXX.XXXXX">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b"/></a>
  <a href="LICENSE">
    <img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-green"/></a>
  <a href="https://scorecard.dev/viewer/?uri=github.com/fsafaei/concerto">
    <img alt="OpenSSF Scorecard" src="https://api.scorecard.dev/projects/github.com/fsafaei/concerto/badge"/></a>
  <a href="https://pypi.org/project/concerto/">
    <img alt="PyPI" src="https://img.shields.io/pypi/v/concerto"/></a>
</p>

---

## Why this exists

Real factories, hospitals, and warehouses already deploy mixed fleets of
robots: a 500&nbsp;Hz industrial arm next to a 50&nbsp;Hz mobile base, a
vision-only manipulator next to a force-feedback one, a vendor&#x2011;A
controller next to a vendor&#x2011;B controller under
**ISO&nbsp;10218-2:2025**. None of these robots were trained together.
At deployment time, your robot's teammate is **opaque** (no policy
access), **heterogeneous** (different morphology and action frequency),
and **ad&nbsp;hoc** (no prior joint training).

Most multi-robot benchmarks assume identical embodiments and shared
training. The few that don't focus on planning or navigation, not on
contact-rich physical manipulation. The intersection of
**Heterogeneity &times; Black-box partner &times; Safety &times; Manipulation** is
empty in the published literature. CHAMBER is built to fill it, and
CONCERTO is the first method designed and certified on it.

### How we sit relative to the closest prior work

Every prior precedent covers at most three of the four aspects. The
table below lists the closest precedent for each pair of aspects; no
published row hits all four. Click any precedent to open the paper.

| Method | Heterogeneous | Black-box partner | Safety bound | Contact-rich manipulation |
|---|:---:|:---:|:---:|:---:|
| [Liu 2024 RSS (LLM&#x2011;AHT)](https://arxiv.org/abs/2406.12224)                            | &#10003; | &#10003; |          |          |
| [COHERENT (LLM&#x2011;MR planning)](https://arxiv.org/abs/2409.15146)                        | &#10003; | &#10003; |          |          |
| [Huriot &amp; Sibai 2025 (conformal CBF)](https://arxiv.org/abs/2409.18862)                  |          | &#10003; | &#10003; |          |
| [HetGPPO](https://arxiv.org/abs/2301.07137) &nbsp;/ [HARL](https://jmlr.org/papers/v25/23-0488.html) (heterogeneous MARL) | &#10003; |          |          |          |
| [Wang et al. 2017 (multi&#x2011;robot CBFs)](https://ieeexplore.ieee.org/document/7989121)   | &#10003; |          | &#10003; |          |
| [RoCoBench (multi&#x2011;robot manipulation)](https://arxiv.org/abs/2307.04738)              | &#10003; |          |          | &#10003; |
| [SafeBimanual (safe bimanual manip.)](https://arxiv.org/abs/2508.18268)                      |          |          | &#10003; | &#10003; |
| **CONCERTO + CHAMBER**                                                                       | **&#10003;** | **&#10003;** | **&#10003;** | **&#10003;** |

Read the table by columns to see what each aspect covers in isolation,
and by rows to see what no single line of work has yet combined.
Contact-rich manipulation appears with multi-robot coordination
(RoCoBench) and with safety (SafeBimanual), but never with black-box
ad-hoc partners under formal safety guarantees at the same time.

See [`adr/ADR-007`](adr/ADR-007-heterogeneity-axis-selection.md) for the
six-axis taxonomy that defines "heterogeneous" precisely, and the
[`docs/explanation/why-aht.md`](docs/explanation/why-aht.md) page for
the long-form positioning.

---

## What's in the box

### Two packages, one repo

**CONCERTO &mdash; the method.** Three-layer safety stack &mdash;
exponential CBF&#x2011;QP backbone
([Wang&#x2011;Ames&#x2011;Egerstedt&nbsp;2017](https://ieeexplore.ieee.org/document/7989121))
&nbsp;+&nbsp; conformal-slack overlay
([Huriot&nbsp;&amp;&nbsp;Sibai&nbsp;2025](https://arxiv.org/abs/2409.18862))
&nbsp;+&nbsp; OSCBF inner filter
([Morton&nbsp;&amp;&nbsp;Pavone&nbsp;2025](https://arxiv.org/abs/2503.17678))
&mdash; plus a frozen-partner ego-AHT training loop (HAPPO variant on a
[HARL](https://github.com/PKU-MARL/HARL) fork). What papers cite.

**CHAMBER &mdash; the benchmark.** ManiSkill&nbsp;v3 wrappers, a
fixed-format communication channel with URLLC&#x2011;anchored degradation
profiles, a partner zoo, evaluation harness, and leaderboard. What users
download and run.

### The six heterogeneity axes CHAMBER measures

| Axis | Symbol | What it varies | Where the priors come from |
|------|--------|----------------|----------------------------|
| Action space            | **AS** | 7&#x2011;DOF arm vs 2&#x2011;DOF mobile base on shared task | HARL, HetGPPO |
| Observation modality    | **OM** | vision-only vs vision + force/torque + proprioception | Visual-tactile peg-in-hole literature |
| Control rate            | **CR** | 500&nbsp;Hz arm vs 50&nbsp;Hz base, chunk size held constant | RTC, A2C2, FAVLA |
| Communication           | **CM** | latency 1&ndash;100&nbsp;ms, jitter &micro;s&ndash;10&nbsp;ms, drop 10<sup>&minus;6</sup>&ndash;10<sup>&minus;2</sup> | 3GPP&nbsp;R17, URLLC |
| Partner familiarity     | **PF** | trained-with vs frozen-novel partner, mid-episode swap | FCP, MEP |
| Safety                  | **SA** | mixed-vendor force-limit / SIL-PL pairs, contact-rich | ISO&nbsp;10218-2:2025 |

Every surviving axis is required to clear a pre-registered
&#8805;20&nbsp;pp homogeneous-vs-heterogeneous gap before it ships in
the v1 benchmark. See
[`adr/ADR-007`](adr/ADR-007-heterogeneity-axis-selection.md) for the
staged Phase&#x2011;0 spike protocol (Stage&nbsp;1: AS&nbsp;+&nbsp;OM,
Stage&nbsp;2: CR&nbsp;+&nbsp;CM, Stage&nbsp;3: PF&nbsp;+&nbsp;SA).

### Repository layout

```text
src/
├── concerto/      # the METHOD  (cite this)
│   ├── safety/    #   exp CBF-QP + conformal overlay + OSCBF + braking fallback
│   ├── training/  #   ego-AHT training loop + deterministic seeding
│   ├── policies/  #   Phase-1 trained checkpoints
│   └── api/       #   public Protocols
└── chamber/       # the BENCHMARK  (run this)
    ├── envs/      #   ManiSkill v3 wrappers
    ├── comm/      #   fixed-format channel + URLLC degradation
    ├── partners/  #   partner zoo (heuristic / frozen-RL / VLA stubs)
    ├── tasks/     #   CHAMBER-Solo / Duo / Quartet (Phase 1+)
    ├── evaluation/#   HRS, pre-registration, leaderboard renderer
    └── benchmarks/#   Stage-0/1/2/3 spike runners

adr/               # 15 Architecture Decision Records (the design rationale)
docs/              # Diátaxis: tutorials / how-to / reference / explanation
tests/             # unit / property / integration / smoke / reproduction
spikes/            # pre-registration YAMLs + result archives
```

---

## Quickstart &nbsp;<sup>30&nbsp;seconds</sup>

```bash
git clone https://github.com/fsafaei/concerto.git
cd concerto
pip install uv && uv sync --group dev

# Smoke test the rig (ADR-001 acceptance criterion).
uv run pytest -m smoke -x -v

# Compose a 5 ms-latency factory-floor channel.
uv run python -c "
from chamber.comm import (
    CommDegradationWrapper, FixedFormatCommChannel, URLLC_3GPP_R17,
)
channel = CommDegradationWrapper(
    FixedFormatCommChannel(),
    URLLC_3GPP_R17['factory'],
    tick_period_ms=1.0,
    root_seed=0,
)
print(channel)
"
```

The six pre-registered URLLC profiles &mdash; `ideal`, `urllc`,
`factory`, `wifi`, `lossy`, `saturation` &mdash; are the Stage&#x2011;2
CM sweep table. See
[`docs/how-to/run-spike.md`](docs/how-to/run-spike.md) for the full
flow.

---

## See it run

<!--
Uncomment the GIF once Stage-0 renders are in assets/.

<p align="center">
  <img src="assets/stage0_smoke.gif"
       alt="CHAMBER Stage-0 smoke task: 3 heterogeneous robots, 100 control steps."
       width="640"/>
  <br/>
  <em>Stage-0 smoke: 3 heterogeneous robots, 100 control steps, real-time.</em>
</p>
-->

A two-screen Stage&#x2011;0 demo (homogeneous baseline vs. heterogeneous
condition with the full safety stack) lands with M5. The reproduction
script &mdash; `scripts/repro/stage0_smoke.sh` &mdash; is committed and
deterministic against `uv.lock` + a `root_seed`.

---

## Leaderboard

<sub>Stage&#x2011;0 acceptance results; updated by
`chamber-render-tables` after each tagged spike.</sub>

| Method | Stage 0 success | Inter-robot collision | Force-limit violation | Conformal &lambda; mean | Reference |
|---|---:|---:|---:|---:|---|
| MAPPO (homogeneous baseline) | _pending_ | _pending_ | _pending_ | _n/a_ | M5 |
| HetGPPO + naive CBF          | _pending_ | _pending_ | _pending_ | _n/a_ | M5 |
| **CONCERTO**                 | _pending_ | _pending_ | _pending_ | _pending_ | M5 |

Submit a new entry: [`docs/how-to/submit-leaderboard.md`](docs/how-to/run-spike.md).

---

## Who is this for

**Multi-robot RL researchers** &mdash; CHAMBER is the first benchmark to
score ad-hoc teamwork at the *manipulation* tier with a measurable
heterogeneity-robustness score (HRS). Start with
[`docs/tutorials/hello-spike.md`](docs/tutorials/hello-spike.md).

**Safe-control researchers** &mdash; CONCERTO's safety module is a
production-grade reference implementation of the
exp&nbsp;CBF&nbsp;+&nbsp;conformal&nbsp;+&nbsp;OSCBF stack with a hard
braking fallback. The unresolved theoretical question
(average-loss&nbsp;&rarr;&nbsp;per-step bound) is documented in
[`adr/ADR-004`](adr/ADR-004-safety-filter.md).

**Robotics practitioners and integrators** &mdash; CHAMBER's
communication profiles are anchored to 3GPP Release&nbsp;17 URLLC and
5G-TSN industrial-trial data, and the safety axis references
ISO&nbsp;10218-2:2025 directly. See
[`docs/explanation/threat-model.md`](docs/explanation/threat-model.md).

---

## Documentation

Full documentation: [**fsafaei.github.io/concerto**](https://fsafaei.github.io/concerto/)

- [**Tutorials**](https://fsafaei.github.io/concerto/latest/tutorials/hello-spike/) &mdash; step-by-step walkthroughs.
- [**How-tos**](https://fsafaei.github.io/concerto/latest/how-to/add-partner/) &mdash; add a partner, add a safety filter, run a spike.
- [**API reference**](https://fsafaei.github.io/concerto/latest/reference/api/) &mdash; generated from docstrings.
- [**ADR index**](https://fsafaei.github.io/concerto/latest/reference/adrs/) &mdash; 15 design decisions with full rationale.
- [**Glossary**](https://fsafaei.github.io/concerto/latest/reference/glossary/) &mdash; HRS, AoI, OSCBF, FCP/MEP, all defined.

---

## Contributing

This is a research project, but it is open from the first commit. We
welcome PRs.

- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for the development flow.
- Look at issues labelled
  [`good-first-issue`](https://github.com/fsafaei/concerto/labels/good-first-issue).
- Sign your commits (`-S`). DCO &nbsp;(`Signed-off-by:`) is required.
- External contributors: the CLA bot will guide you on first PR.
- Every PR cites the ADR section it touches (e.g. `ADR-004 &sect;6.2`).
  We treat the ADRs as the design contract; if your PR motivates a
  change to them, propose a new ADR rather than editing an Accepted one.

Code of Conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
Security policy: [`SECURITY.md`](SECURITY.md).

---

## Citing CONCERTO &amp; CHAMBER

If you use CONCERTO or CHAMBER in your research, please cite the
preprint. Until the preprint is on arXiv (target: 2026&#x2011;06), cite
the archived software release via its Zenodo DOI:

```bibtex
@software{safaei2026concerto,
  author       = {Safaei, Farhad},
  title        = {{CONCERTO} and {CHAMBER}: Contact-rich Coordination
                  with Opaque, Heterogeneous Teammates},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.20128469},
  url          = {https://doi.org/10.5281/zenodo.20128469},
  note         = {arXiv preprint forthcoming},
}
```

Citation entries are also in [`CITATION.cff`](CITATION.cff) so GitHub
renders a "Cite this repository" button.

---

## Acknowledgments

CONCERTO builds on
[Wang&nbsp;et&nbsp;al.&nbsp;2017](https://ieeexplore.ieee.org/document/7989121)
(exponential CBFs),
[Huriot&nbsp;&amp;&nbsp;Sibai&nbsp;2025](https://arxiv.org/abs/2409.18862)
(conformal CBFs), and
[Morton&nbsp;&amp;&nbsp;Pavone&nbsp;2025](https://arxiv.org/abs/2503.17678)
(OSCBF). CHAMBER is a wrapper layer over
[ManiSkill&nbsp;v3](https://github.com/haosulab/ManiSkill) and depends
on a fork of
[HARL](https://github.com/PKU-MARL/HARL) for the training stack.
Acknowledgment of any contributions and corrections welcome.

---

## License

Apache 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
The full Software Bill of Materials is at
[`sbom.spdx.json`](sbom.spdx.json) and is regenerated on every release.
