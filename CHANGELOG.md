# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is on `0.x`, MINOR-version bumps may break the public API
per SemVer §4.

## [Unreleased]

### Added

- **M2 — Communication stack** (ADR-003, ADR-006).
  - `chamber.comm` public API: `CommChannel` Protocol, `CommPacket` /
    `Pose` / `TaskStatePredicate` TypedDicts, `SCHEMA_VERSION` constant,
    `ChamberCommError`, `ChamberCommQPSaturationWarning`.
  - `FixedFormatCommChannel` — schema-versioned encode/decode with a
    round-trip guarantee on the typed subset of state.
  - `AoIClock` — per-uid Age-of-Information accounting (Ballotta &
    Talak 2024 semantics).
  - `LearnedOverlay` — opt-in null-safe slot for jointly-trained B6
    partners (HetGPPO / CommFormer); Phase-1 fills in the encoder.
  - `CommDegradationWrapper` + `DegradationProfile` + the six named
    `URLLC_3GPP_R17` profiles (ideal / urllc / factory / wifi / lossy /
    saturation), anchored to URLLC + 3GPP Release 17 industrial-trial
    data. Per-step Bernoulli drop, clipped-Normal latency queue,
    AoI-aware delivery, telemetry counters via `CommDegradationStats`.
  - `saturation_guard` — fires `ChamberCommQPSaturationWarning` only
    under the `saturation` profile or when the QP solver exceeds the
    1 ms OSCBF target (ADR-004), pinning the contract for M3.
  - `concerto.training.seeding.derive_substream` — deterministic
    BLAKE2b-keyed seeding harness (P6) the comm channel and degradation
    wrapper draw from.
  - `concerto.safety.solve_qp_stub` — no-op QP placeholder so the
    saturation property test exists from M2 (M3 replaces it).
  - Stage-0 smoke (`chamber.benchmarks.stage0_smoke`) now composes
    `CommDegradationWrapper(FixedFormatCommChannel(), URLLC_3GPP_R17["factory"])`;
    the M1 stub channel is retired.
- **M1 — Platform stack** (ADR-001, ADR-005).
  - Three wrappers above ManiSkill v3 (`PerAgentActionRepeatWrapper`,
    `TextureFilterObsWrapper`, `CommShapingWrapper`) totalling under
    230 LOC, with `ChamberEnvCompatibilityError` for loud-fail under
    unsupported envs.
  - Stage-0 smoke env (`make_stage0_env`) reproducing the ADR-001
    3-robot acceptance scenario; CPU-only wrapper-structure tests cover
    conditions (a)/(b)/(c) and the Vulkan tier auto-skips when SAPIEN
    is unavailable.
- Repo scaffolding (M0): packaging, CI, docs, licence, hygiene scripts.
- Two top-level packages: `concerto` (METHOD) and `chamber` (BENCHMARK).

[Unreleased]: https://github.com/fsafaei/concerto/compare/v0.0.1.dev0...HEAD
