# Positioning — why this exists

Real factories already pair robots that were never trained together.
A 500 Hz industrial arm next to a 50 Hz mobile base; a vision-only
manipulator next to a force-feedback one; a vendor-A controller next
to a vendor-B controller under binding **ISO 10218-2:2025**. At
deployment time, your robot's teammate is **opaque** (no policy
access), **heterogeneous** (different morphology and action
frequency), and **ad hoc** (no prior joint training). Hospitals and
warehouses are the same picture.

Most multi-robot benchmarks assume identical embodiments and shared
training. The few that don't focus on planning or navigation, not on
contact-rich physical manipulation. The intersection of
**Heterogeneity × Black-box partner × Safety × Manipulation** is
empty in the published literature. CHAMBER is built to fill it, and
CONCERTO is the first method designed against this four-aspect
contract.

## How we sit relative to the closest prior work

Every prior precedent covers at most three of the four aspects. The
table lists the closest precedent for each pair of aspects; no
published row hits all four.

| Method | Heterogeneous | Black-box partner | Safety bound | Contact-rich manipulation |
|---|:---:|:---:|:---:|:---:|
| [Liu 2024 RSS (LLM-AHT)](https://arxiv.org/abs/2406.12224)                            | ✓ | ✓ |   |   |
| [COHERENT (LLM-MR planning)](https://arxiv.org/abs/2409.15146)                        | ✓ | ✓ |   |   |
| [Huriot & Sibai 2025 (conformal CBF)](https://arxiv.org/abs/2409.18862)               |   | ✓ | ✓ |   |
| [HetGPPO](https://arxiv.org/abs/2301.07137) / [HARL](https://jmlr.org/papers/v25/23-0488.html) (heterogeneous MARL) | ✓ |   |   |   |
| [Wang et al. 2017 (multi-robot CBFs)](https://ieeexplore.ieee.org/document/7989121)   | ✓ |   | ✓ |   |
| [RoCoBench (multi-robot manipulation)](https://arxiv.org/abs/2307.04738)              | ✓ |   |   | ✓ |
| [SafeBimanual (safe bimanual manip.)](https://arxiv.org/abs/2508.18268)               |   |   | ✓ | ✓ |
| **CONCERTO + CHAMBER**                                                                | **✓** | **✓** | **✓** | **✓** |

Read the table by columns to see what each aspect covers in
isolation, and by rows to see what no single line of work has yet
combined. Contact-rich manipulation appears with multi-robot
coordination (RoCoBench) and with safety (SafeBimanual), but never
with black-box ad-hoc partners under explicit safety assumptions at
the same time. *Heterogeneous* here is the four-aspect literature-gap
level; CHAMBER's six measurable sub-axes decompose it further (below).
See [Why heterogeneous ad-hoc teamwork?](why-aht.md) for the
long-form argument.

## The six heterogeneity axes

| Axis | Symbol | What it varies | Validity (ADR-026) | Where the priors come from |
|------|--------|----------------|--------------------|----------------------------|
| Action space            | **AS** | 7-DOF arm vs 2-DOF mobile base on shared task | construct-invalid on pick-place → retained as control; null on co-carry | HARL, HetGPPO |
| Observation modality    | **OM** | vision-only vs vision + force/torque + proprioception | construct-invalid on pick-place → retained as control | Visual-tactile peg-in-hole literature |
| Control rate            | **CR** | 500 Hz arm vs 50 Hz base, chunk size held constant | untested | RTC, A2C2, FAVLA |
| Communication           | **CM** | latency 1–100 ms, jitter µs–10 ms, drop 10⁻⁶–10⁻² | untested | 3GPP R17, URLLC |
| Partner familiarity     | **PF** | trained-with vs frozen-novel partner, mid-episode swap | untested | FCP, MEP |
| Safety                  | **SA** | mixed-vendor force-limit / SIL-PL pairs, contact-rich | untested | ISO 10218-2:2025 |

A preregistered ≥20 percentage-point homogeneous-vs-heterogeneous gap
is necessary but not sufficient: an axis ships in the benchmark only
on a task meeting the coupling-validity criterion (ADR-026), under
the CHAMBER-Bench v1.0 protocol (ADR-027). The historical staged
spike protocol (Stage 1: AS + OM, Stage 2: CR + CM, Stage 3: PF + SA)
is ADR-007.

## Who this is for

**Multi-robot RL researchers** — CHAMBER scores ad-hoc teamwork at
the *manipulation* tier with verifiable, preregistered result
bundles. Start with the
[five-minute quickstart](https://github.com/fsafaei/concerto#five-minute-quickstart).

**Safe-control researchers** — CONCERTO's safety module is a
reference implementation of the exponential-CBF + conformal + OSCBF
stack with a hard braking fallback. The unresolved theoretical
question (average-loss → per-step bound) is documented in ADR-004.

**Robotics practitioners and integrators** — CHAMBER's communication
profiles are anchored to 3GPP Release-17 URLLC and 5G-TSN
industrial-trial data, and the safety axis references
ISO 10218-2:2025 directly. See the [threat model](threat-model.md).

## Non-goals

CHAMBER is **not** a navigation, planning, or generic reinforcement-
learning benchmark; the four-aspect intersection requires
contact-rich physical manipulation. CONCERTO is **not** a certified
safety product — it is a research-grade reference implementation and
is **not** a substitute for safety engineering in production
deployments.

## Roadmap

The project advances in phases. Phase 0 locked the design contract
(the ADRs) and ran the staged heterogeneity-axis spikes; its outcome
was the coupling-validity reinterpretation (ADR-026). Phase 1 shipped
the CHAMBER-Bench v1.0 protocol (ADR-027), the admission campaign,
the versioned partner zoo, and the populated per-task leaderboard.
Ahead: heterogeneity axes that survive coupling-valid
operationalization (control rate and communication are the next
candidates), additional Tier-2 task admissions from the Tier-3
candidate list, and — later — a real-robot demonstration platform.

Day-to-day progress: the
[CHANGELOG](https://github.com/fsafaei/concerto/blob/main/CHANGELOG.md)
and the [issues board](https://github.com/fsafaei/concerto/issues).
