# FAQ

## How does CHAMBER differ from RoCoBench, SafeBimanual, or BiGym?

RoCoBench covers Heterogeneity × Manipulation on MuJoCo with
multi-arm LLM-dialectic coordination but does not address black-box
partners or formal safety bounds. SafeBimanual covers
Safety × Manipulation on a single bimanual platform. BiGym is
single-embodiment. CHAMBER targets the four-aspect intersection
(Heterogeneity × Black-box × Safety × Manipulation) at the
*substrate* level — thin wrapper layers above ManiSkill v3, a
fixed-format communication stack, and a partner zoo — rather than as
a curated task set. See ADR-001 and ADR-005 for the simulator-base
decision, and [Positioning](positioning.md) for the full prior-work
table.

## Is the safety guarantee per-step or asymptotic?

The conformal slack overlay (Huriot & Sibai 2025, Theorem 3) gives a
distribution-free ε + o(1) **long-term average-loss** bound, not a
per-step bound. For contact-rich manipulation where a single
violation can be irreversible, the hard braking fallback
(Wang, Ames & Egerstedt 2017, eq. 17) is the per-step backstop.
Sharpening the average-loss bound to per-step is the project's
headline open theoretical question; see ADR-004 §Open questions.

## Can I plug in my own partner or safety filter?

Yes. Partners implement the `FrozenPartner` Protocol in
`chamber.partners.api` and register with the `@register_partner`
decorator; see [Add a partner](../how-to/add-partner.md). Safety
filters implement the `SafetyFilter` Protocol in
`concerto.safety.api`; see [Add a filter](../how-to/add-filter.md).
To put a method on the leaderboard, follow
[Submit a leaderboard entry](../how-to/submit-leaderboard.md).

## What does "admitted task" mean, and why is pick-and-place not one?

A task enters the scored tier only by a committed admission report
showing, under preregistered thresholds, that a reference pair solves
it (A1), the best single robot cannot (A2), and cutting the coupling
channel collapses performance (A3). Pick-and-place passed A1 but
failed A2 and A3 — one arm solves it alone — so it is retained as a
Tier-1 control. See the
[evaluation protocol](evaluation-protocol.md).

## What's the relationship between CONCERTO and CHAMBER?

CONCERTO is the method (safety stack + ego ad-hoc-teamwork training);
CHAMBER is the benchmark (env wrappers + comm + partner zoo + tasks +
evaluation). Two top-level packages in one wheel, with a one-way
dependency: `chamber → concerto`. Canonical sentence: *we evaluate
CONCERTO on CHAMBER.*

## Is this reproducible bit-for-bit?

CPU runs are byte-identical under the committed `uv.lock` plus a
`root_seed`, via the determinism harness in
`concerto.training.seeding`. This is what makes `chamber-eval verify`
able to recompute a committed bundle's summary statistics exactly.
GPU runs are deterministic only up to CUDA non-determinism in PyTorch
reductions; the multi-seed aggregate metrics in
[Evaluation](../reference/evaluation.md) are the canonical way to
compare across seeds.

## Why ManiSkill v3 and not Isaac Lab?

ADR-001's contingent rule was "extend the simulator if its
abstractions admit the heterogeneity-axis controls without
monkey-patching." ManiSkill v3 passes that test at roughly 230 lines
of wrappers; Isaac Lab would have required a months-long standalone
build. Isaac Lab remains a viable secondary path if upstream API
constraints force a migration — the env-adapter layer is
intentionally thin. See ADR-001 and ADR-005.
