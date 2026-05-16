# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is on `0.x`, MINOR-version bumps may break the public API
per SemVer §4.

## [0.3.1](https://github.com/fsafaei/concerto/compare/v0.3.0...v0.3.1) (2026-05-16)


### Bug Fixes

* **ci:** correct uv TestPyPI publish target + add workflow_dispatch for re-fires ([#128](https://github.com/fsafaei/concerto/issues/128)) ([8d49566](https://github.com/fsafaei/concerto/commit/8d495662d3baef45ff4dc04d726f62785d217a6f))

## [0.3.0](https://github.com/fsafaei/concerto/compare/v0.2.0...v0.3.0) (2026-05-16)


### Features

* **benchmarks:** EgoActionFactory Protocol + Tier-1 contract tests for the Stage-1 seam (plan/07 §T5b.2) ([#123](https://github.com/fsafaei/concerto/issues/123)) ([04be156](https://github.com/fsafaei/concerto/commit/04be156d05ef71ab3104c0ec42521020c6290709))
* **cli:** chamber-spike summarize-month3 — Month-3 lock-priority report (plan/07 §T5b.10) ([#124](https://github.com/fsafaei/concerto/issues/124)) ([e06bda9](https://github.com/fsafaei/concerto/commit/e06bda93588bbc7db89f28fb28c0dff0ddd746fb))
* **cli:** chamber-spike verify-prereg --all (plan/07 §6 [#5](https://github.com/fsafaei/concerto/issues/5)) ([#115](https://github.com/fsafaei/concerto/issues/115)) ([724e2bd](https://github.com/fsafaei/concerto/commit/724e2bdac349dfb778b0376f2aa3e3691402cdc6))
* **training,adr:** reject non-frozen partner at trainer construction + amend ADR-002 §Risks [#1](https://github.com/fsafaei/concerto/issues/1) for slope test ([#119](https://github.com/fsafaei/concerto/issues/119)) ([ec57756](https://github.com/fsafaei/concerto/commit/ec5775649c62c6e55725ff517f569e869b7957c7))


### Bug Fixes

* **benchmarks:** OM tuple-collision in stage1_om — homo/hetero condition shape divergence ([#122](https://github.com/fsafaei/concerto/issues/122)) ([73070a1](https://github.com/fsafaei/concerto/commit/73070a1d2fa2023c22f0818dd5c55427b2594255))


### Documentation

* **adr-index:** ADR-002 §Risks [#1](https://github.com/fsafaei/concerto/issues/1) slope-test amendment open-work flag ([#120](https://github.com/fsafaei/concerto/issues/120)) ([3933c38](https://github.com/fsafaei/concerto/commit/3933c38e8eaa08955e4f721627ab5efc33d491db))
* **adr:** ADR-007 revision 4 — introduce Stage 1a/1b sub-stages with Phase-1 trigger ([#121](https://github.com/fsafaei/concerto/issues/121)) ([06aaa8d](https://github.com/fsafaei/concerto/commit/06aaa8da9265711b018a55fa570bec19e44cad03))

## [0.2.0](https://github.com/fsafaei/concerto/compare/v0.1.1...v0.2.0) (2026-05-15)


### Features

* **benchmarks:** Stage-1 AS adapter (T5b.2 a; plan/07 §3) ([#112](https://github.com/fsafaei/concerto/issues/112)) ([3e47075](https://github.com/fsafaei/concerto/commit/3e47075fbdcf579341a13e92492a0a29c76e4381))
* **benchmarks:** Stage-1 OM adapter (T5b.2 b; plan/07 §3; closes M5b) ([#113](https://github.com/fsafaei/concerto/issues/113)) ([c93a833](https://github.com/fsafaei/concerto/commit/c93a83316f4028dff078b42a9664dbc6d9368dec))
* **benchmarks:** stage0_smoke env adapter for ego-AHT training (T4b.3) ([#74](https://github.com/fsafaei/concerto/issues/74)) ([c56df38](https://github.com/fsafaei/concerto/commit/c56df38ee2fefd59d31844ff55d3ff8d3e652378))
* **chamber:** chamber-eval accepts multiple SpikeRun files for multi-axis HRS ([#99](https://github.com/fsafaei/concerto/issues/99)) ([d2398ae](https://github.com/fsafaei/concerto/commit/d2398ae13b01264d8c56fea8b1585f29887a1809))
* **chamber:** minimum evaluation spine — results schema, preregistration loader, hierarchical bootstrap, HRS-vector + scalar, three-table renderer ([#94](https://github.com/fsafaei/concerto/issues/94)) ([24d3a61](https://github.com/fsafaei/concerto/commit/24d3a610fac5b598043255a00aa8f093c5896c93))
* **cli:** chamber-spike run + next-stage gate (T5b.1; plan/07 §5) ([#111](https://github.com/fsafaei/concerto/issues/111)) ([280b46a](https://github.com/fsafaei/concerto/commit/280b46a2eade6a3cca202b2116a6cf09013ae0e2))
* **cli:** chamber-spike verify-prereg + list-axes + list-profiles (T5b.1) ([#110](https://github.com/fsafaei/concerto/issues/110)) ([5e4e12b](https://github.com/fsafaei/concerto/commit/5e4e12bf646126566fc4430bfdc7bbf032117b78))
* **envs,partners:** draft-zoo Stage-0 round-trip — closes M4 gate (T4.9) ([#105](https://github.com/fsafaei/concerto/issues/105)) ([72bbab8](https://github.com/fsafaei/concerto/commit/72bbab8e40701d1f122479913db248ad24811659))
* **eval:** enforce iid-not-allowed-for-leaderboard bootstrap policy in PreregistrationSpec ([#100](https://github.com/fsafaei/concerto/issues/100)) ([dc69217](https://github.com/fsafaei/concerto/commit/dc6921799f813ecca37eae52e32faeb779799440))
* **partners:** FrozenHARLPartner adapter (T4.6) ([#104](https://github.com/fsafaei/concerto/issues/104)) ([3f87e72](https://github.com/fsafaei/concerto/commit/3f87e7217bee20550ce8856610982929def7da75))
* **partners:** FrozenMAPPOPartner adapter (T4.5) ([#103](https://github.com/fsafaei/concerto/issues/103)) ([1eab560](https://github.com/fsafaei/concerto/commit/1eab560f195cbc4983160a75e709972112acd4b2))
* **safety:** per-agent control models, ego-only mode, and heterogeneous-action CBF-QP ([#93](https://github.com/fsafaei/concerto/issues/93)) ([d8a8a11](https://github.com/fsafaei/concerto/commit/d8a8a1157cef51868b1edb49d0102062f5d4ffe9))
* **safety:** split SafetyFilter into EgoOnlySafetyFilter and JointSafetyFilter Protocols ([#97](https://github.com/fsafaei/concerto/issues/97)) ([e100c4b](https://github.com/fsafaei/concerto/commit/e100c4bc95f7c8364a2f8b7b47c5d7678d554e03))
* **spikes:** six ADR-007 pre-registration YAMLs (T5a.7) ([#109](https://github.com/fsafaei/concerto/issues/109)) ([153a67d](https://github.com/fsafaei/concerto/commit/153a67de04a583b3689f9ff284c2099209248f06))
* **training,cli:** runtime device + chamber-spike train + GPU host kit (T4b.14 prep) ([#76](https://github.com/fsafaei/concerto/issues/76)) ([0a1b0ca](https://github.com/fsafaei/concerto/commit/0a1b0ca8952ad0e4784d3d888d18c8e5c096eee7))
* **training:** empirical-guarantee scaffolding + xfail trip-wire probe (T4b.13) ([#63](https://github.com/fsafaei/concerto/issues/63)) ([48f46e9](https://github.com/fsafaei/concerto/commit/48f46e92e5d2e0175281392773e13c7753060274))
* **training:** replace empirical-guarantee statistic with slope test (closes [#62](https://github.com/fsafaei/concerto/issues/62)) ([#72](https://github.com/fsafaei/concerto/issues/72)) ([75b5c7a](https://github.com/fsafaei/concerto/commit/75b5c7a69d84e127b0eb0da9cf39ae9f2b3e4a2f))


### Bug Fixes

* **deps:** bump torch to &gt;=2.8,&lt;2.9 to close three CVEs ([#81](https://github.com/fsafaei/concerto/issues/81)) ([633e4b9](https://github.com/fsafaei/concerto/commit/633e4b90d49c81914d4fd9fc517d7f4facba91d2))
* **envs,docker:** make zoo-seed run on WSL2 + Docker Engine ([#107](https://github.com/fsafaei/concerto/issues/107)) ([1113e23](https://github.com/fsafaei/concerto/commit/1113e23e8be91338b80b3c736458b9fa9994a61f))
* **readme:** correct Quickstart to use encode/decode and add an executable docs test ([#89](https://github.com/fsafaei/concerto/issues/89)) ([4711006](https://github.com/fsafaei/concerto/commit/47110069c779a44ecfa64ee1d9e3162acdba1408))
* **safety:** embodiment-aware multi-pair braking fallback ([#91](https://github.com/fsafaei/concerto/issues/91)) ([854cfe1](https://github.com/fsafaei/concerto/commit/854cfe1d9b5fe1677d6e8fe389149c4da0b490b5))
* **safety:** separate conformal prediction-gap loss from constraint-violation signal ([#90](https://github.com/fsafaei/concerto/issues/90)) ([786baa1](https://github.com/fsafaei/concerto/commit/786baa1408b44c4f6c134cafad2d0b53a3675b49))
* **training:** handle time-limit truncation in GAE bootstrap (issue [#62](https://github.com/fsafaei/concerto/issues/62) root cause) ([#70](https://github.com/fsafaei/concerto/issues/70)) ([e0819be](https://github.com/fsafaei/concerto/commit/e0819be91d0200d05243018fb2f7fcd3f07fae0b))


### Documentation

* **adr-003+006:** cite TSN clause-level standards underneath URLLC ([#84](https://github.com/fsafaei/concerto/issues/84)) ([caec8c5](https://github.com/fsafaei/concerto/commit/caec8c5144389e7206855ed5dc0c4c483021a1f4))
* **adr-004:** anchor safety filter in foundational CBF and CP canon ([#83](https://github.com/fsafaei/concerto/issues/83)) ([a1951ec](https://github.com/fsafaei/concerto/commit/a1951ec8d9a03815bbc0115b94a48b90bde82baf))
* **adr-014:** add machine functional-safety stack and rliable reporting contract ([#85](https://github.com/fsafaei/concerto/issues/85)) ([8b60c13](https://github.com/fsafaei/concerto/commit/8b60c13f309462a1596a8b824d4522ac41b3c5e4))
* **brand:** crop logo PNGs to their content bounding box ([#69](https://github.com/fsafaei/concerto/issues/69)) ([f42203b](https://github.com/fsafaei/concerto/commit/f42203beab6b9761b46b1d056e16552fd7546ec3))
* **brand:** use the white-text logo + size it to fit the docs header ([#68](https://github.com/fsafaei/concerto/issues/68)) ([2cde7d7](https://github.com/fsafaei/concerto/commit/2cde7d7bf690a3daf534a1a7aeb1032392d58505))
* **brand:** wire up the CONCERTO logo for README + mkdocs theme ([#67](https://github.com/fsafaei/concerto/issues/67)) ([0a1e2f7](https://github.com/fsafaei/concerto/commit/0a1e2f7ad06191870bfdbba0811ed26dfd147bd3))
* **changelog:** remove stale hand-written [Unreleased] block ([#87](https://github.com/fsafaei/concerto/issues/87)) ([19a3e58](https://github.com/fsafaei/concerto/commit/19a3e58c84bcda05b8ca2409c2d52ba10e14b9b2))
* **governance:** introduce ADR status taxonomy and downgrade README to Provisional language ([#92](https://github.com/fsafaei/concerto/issues/92)) ([181cfa2](https://github.com/fsafaei/concerto/commit/181cfa2a20e33c1388982e55e7711bd760788c1c))
* **readme:** add Zenodo DOI badge to the top of the badge row ([#61](https://github.com/fsafaei/concerto/issues/61)) ([718c14f](https://github.com/fsafaei/concerto/commit/718c14faf3fdb80a1858d94fce6a0b7a4bfc4c7a))
* **readme:** correct comparison table — purge Singh row, add precedent links ([#75](https://github.com/fsafaei/concerto/issues/75)) ([48a59e2](https://github.com/fsafaei/concerto/commit/48a59e2b19694e516466af79f201fbba479c78b9))
* **readme:** downgrade safety wording from "certified" / "high-probability" to match ADR-004 Provisional status ([#96](https://github.com/fsafaei/concerto/issues/96)) ([318824b](https://github.com/fsafaei/concerto/commit/318824b216e6b6111129569aa4189c53fb793a1a))
* **readme:** modernise structure, add status / TL;DR / TOC / FAQ / roadmap / non-goals / stability sections ([#86](https://github.com/fsafaei/concerto/issues/86)) ([64f7e5a](https://github.com/fsafaei/concerto/commit/64f7e5a2d81ef867c9f082512afd981638a0a4e5))
* **readme:** use ``/latest/`` versioned path for docs deep-links ([#65](https://github.com/fsafaei/concerto/issues/65)) ([1b6cf88](https://github.com/fsafaei/concerto/commit/1b6cf884a65cd6937540f56ac8f321b635957bb1))
* **readme:** wire up the Zenodo DOI in CITATION.cff and the bibtex ([#66](https://github.com/fsafaei/concerto/issues/66)) ([1c4a82c](https://github.com/fsafaei/concerto/commit/1c4a82c8200327c4630e9c591db2ff7f6d2513d8))
* **reference:** add canonical refs.bib and normalise public citation shorthand ([#88](https://github.com/fsafaei/concerto/issues/88)) ([eb6dc5e](https://github.com/fsafaei/concerto/commit/eb6dc5e9fa97b966145e8315a1a4ae9a3109bc80))
* **reference:** add public literature, standards, and evaluation pages ([#82](https://github.com/fsafaei/concerto/issues/82)) ([fb9367b](https://github.com/fsafaei/concerto/commit/fb9367b9db7d2b990e5868724db7652dbbb4c4a1))
* **run-on-gpu:** clarify that slope=0 on stage0_smoke is the expected rig-validation outcome ([#95](https://github.com/fsafaei/concerto/issues/95)) ([2134613](https://github.com/fsafaei/concerto/commit/2134613c8f7f63306c9aa4b42a5681ae1fea31b8))

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
