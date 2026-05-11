# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is on `0.x`, MINOR-version bumps may break the public API
per SemVer §4.

## [0.1.1](https://github.com/fsafaei/concerto/compare/v0.1.0...v0.1.1) (2026-05-11)


### Bug Fixes

* **release:** sync __version__ with pyproject + grant publish write perms ([#58](https://github.com/fsafaei/concerto/issues/58)) ([735d2db](https://github.com/fsafaei/concerto/commit/735d2db1cd1c08b3b02f9e398ccf4834ef25ee2e))

## [0.1.0](https://github.com/fsafaei/concerto/compare/v0.0.1...v0.1.0) (2026-05-11)


### Features

* **deps:** bump HARL fork to v0.1.0-aht (M4b-7) ([#38](https://github.com/fsafaei/concerto/issues/38)) ([489bbab](https://github.com/fsafaei/concerto/commit/489bbab26f2034de9dc09612532d89045ea3a9ec))
* **deps:** pin HARL fork at v0.0.0-vendored (T4b.2) ([#36](https://github.com/fsafaei/concerto/issues/36)) ([bf3847e](https://github.com/fsafaei/concerto/commit/bf3847e763b3fc8d7b06135d075c0436be8040b4))
* **partners:** add FrozenPartner Protocol + PartnerSpec dataclass (T4.1) ([#25](https://github.com/fsafaei/concerto/issues/25)) ([9e6541e](https://github.com/fsafaei/concerto/commit/9e6541e044e70cff23f67292b099313718a60e47))
* **partners:** add OpenVLA + CrossFormer Phase-1 stubs (T4.7) ([#28](https://github.com/fsafaei/concerto/issues/28)) ([6aa1d3a](https://github.com/fsafaei/concerto/commit/6aa1d3a5c60a2d347e7d8b9082fa4db9f1012ead))
* **partners:** add registry decorator + PartnerBase shield (T4.2 + T4.3) ([#26](https://github.com/fsafaei/concerto/issues/26)) ([02c5f31](https://github.com/fsafaei/concerto/commit/02c5f3114efa1651638a880d7f2653b155cf31ce))
* **partners:** add ScriptedHeuristicPartner + Phase-0 draft zoo (T4.4 + T4.8) ([#27](https://github.com/fsafaei/concerto/issues/27)) ([1639e90](https://github.com/fsafaei/concerto/commit/1639e90ea63c73fe81d15bf8885c2685ebeda330))
* **safety:** add hard braking fallback (T3.7) ([#17](https://github.com/fsafaei/concerto/issues/17)) ([c367cde](https://github.com/fsafaei/concerto/commit/c367cde0a2a85b00c1b97cc3243e0c28975ab22d))
* **safety:** add OSCBF two-level QP inner filter (T3.8) ([#18](https://github.com/fsafaei/concerto/issues/18)) ([a096da3](https://github.com/fsafaei/concerto/commit/a096da387cc677b36aa424d8ffbde9adc1a4711a))
* **safety:** add three-table safety report emitter (T3.9; ADR-014) ([#19](https://github.com/fsafaei/concerto/issues/19)) ([db13146](https://github.com/fsafaei/concerto/commit/db131463d3ff241b2ee137aea5b9d72942df9f7c))
* **safety:** replace M2 solve_qp_stub with real Clarabel benchmark (T3.10) ([#20](https://github.com/fsafaei/concerto/issues/20)) ([63bd6ec](https://github.com/fsafaei/concerto/commit/63bd6ec39f46a1ccacaa6b6936ae0311f8674a15))
* **scripts:** add HARL fork creation recipe + patches (T4b.1, T4b.4-T4b.7) ([#35](https://github.com/fsafaei/concerto/issues/35)) ([2528985](https://github.com/fsafaei/concerto/commit/25289858e0e63213cf68119c4dfea3967ba48347))
* **training:** add checkpoint save/load + SHA-256 integrity (T4b.12) ([#31](https://github.com/fsafaei/concerto/issues/31)) ([3a6e9a9](https://github.com/fsafaei/concerto/commit/3a6e9a963e1a7ac913e0650f108e01659cc3214b))
* **training:** add ego-AHT train() shell + chamber-side runner (T4b.11) ([#34](https://github.com/fsafaei/concerto/issues/34)) ([b9122a4](https://github.com/fsafaei/concerto/commit/b9122a457e25b128f963474091a891cd9b571e5a))
* **training:** add Hydra config scaffolding + EgoAHTConfig (plan/05 §2) ([#33](https://github.com/fsafaei/concerto/issues/33)) ([c0c3f92](https://github.com/fsafaei/concerto/commit/c0c3f92d8d65f343f97e1d2b50e0b379eabcca4e))
* **training:** add structured logging for ego-AHT runs (T4b.10) ([#30](https://github.com/fsafaei/concerto/issues/30)) ([046ba42](https://github.com/fsafaei/concerto/commit/046ba42989dbdc4a1310cd5b295c0f749d7b2606))
* **training:** EgoPPOTrainer + GAE property test (M4b-8a) ([#44](https://github.com/fsafaei/concerto/issues/44)) ([89c577b](https://github.com/fsafaei/concerto/commit/89c577b63ec63b3cb53e7458897ad6eae9c69ad8))


### Bug Fixes

* align axis count with ADR-007 revision 3 (four→six evaluation axes) ([#10](https://github.com/fsafaei/concerto/issues/10)) ([2c4b102](https://github.com/fsafaei/concerto/commit/2c4b1027f4ae9474decc05aaf33fd107f4a4bae7))
* **ci:** run OSV-Scanner via CLI to avoid SARIF/Code-Scanning gate ([#23](https://github.com/fsafaei/concerto/issues/23)) ([af2cca7](https://github.com/fsafaei/concerto/commit/af2cca734e44742cf0c7142a81dfb899e64f61b7))


### Documentation

* correct GitHub URLs and fix gh-pages root redirect ([#54](https://github.com/fsafaei/concerto/issues/54)) ([68a93cf](https://github.com/fsafaei/concerto/commit/68a93cf9eaec16aca571456538effd73999abee9))
* **partners:** add how-to/add-partner.md walkthrough + code-block test (T4.10) ([#29](https://github.com/fsafaei/concerto/issues/29)) ([76ca9bd](https://github.com/fsafaei/concerto/commit/76ca9bd6707c8d73e406736ddbd2015d12b18381))
* **readme:** rewrite README as researcher-first benchmark landing page ([#55](https://github.com/fsafaei/concerto/issues/55)) ([3ce9eea](https://github.com/fsafaei/concerto/commit/3ce9eea02c943efea7c8eb75ddd0c3f27665e39e))
* **safety:** expand why-conformal walkthrough + add code-block tests (T3.12) ([#22](https://github.com/fsafaei/concerto/issues/22)) ([1139f4e](https://github.com/fsafaei/concerto/commit/1139f4e3283d6f1ba64d17425e9722324bde81be))
* **tutorials:** replace hello-spike placeholder with M4b 1k-frame demo (T4b.15 partial) ([#37](https://github.com/fsafaei/concerto/issues/37)) ([ceefb1b](https://github.com/fsafaei/concerto/commit/ceefb1b594fef222b39df3776a9f8d7a5b81760a))

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
