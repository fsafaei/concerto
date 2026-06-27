# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the project is on `0.x`, MINOR-version bumps may break the public API
per SemVer §4.

## [0.8.0](https://github.com/fsafaei/concerto/compare/v0.7.0...v0.8.0) (2026-06-27)


### Features

* **benchmarks:** TrainedPolicyFactory — Phase-1 EgoActionFactory wiring concerto.training.ego_aht.train (ADR-007 §Stage 1b; plan/07 §T5b.2) ([#162](https://github.com/fsafaei/concerto/issues/162)) ([c436a80](https://github.com/fsafaei/concerto/commit/c436a8073e3f934315513b5c6db6f50fae0d06c5))
* **cocarry:** Rung-2 frozen learned incumbent (training + freeze) per ADR-026 ([#246](https://github.com/fsafaei/concerto/issues/246)) ([41ddcc7](https://github.com/fsafaei/concerto/commit/41ddcc7d5d4da7faa025db1a86779dc46bdee167))
* **cocarry:** Rung-3 policy-heterogeneity (PH) measurement — pooled null, one stiff-teammate drop ([#248](https://github.com/fsafaei/concerto/issues/248)) ([7e5b88a](https://github.com/fsafaei/concerto/commit/7e5b88a4f286cfc25dbbaf66b1fc698fc2d3f4fc))
* **cocarry:** Rung-4 embodiment-heterogeneity (EH) capstone — feasibility finding ([#249](https://github.com/fsafaei/concerto/issues/249)) ([2f6098e](https://github.com/fsafaei/concerto/commit/2f6098e62853ff15e35b2e337400e796f6059995))
* **cocarry:** Rung-4b compliant coupling — over-coupling resolved; EH blocked by embodiment-biased stress proxy ([#251](https://github.com/fsafaei/concerto/issues/251)) ([b01dab3](https://github.com/fsafaei/concerto/commit/b01dab3b6185734d85e2eb2ab5aaa99214fa74f3))
* **cocarry:** Rung-4c invariant stress instrument — embodiment genuinely degrades cooperation (qualifying EH drop) ([#252](https://github.com/fsafaei/concerto/issues/252)) ([f12f62d](https://github.com/fsafaei/concerto/commit/f12f62d4139dc90cfffcb1e9129432a8b45aad65))
* **deps:** consume harl-aht from PyPI; drop train-group git+URL ([#173](https://github.com/fsafaei/concerto/issues/173)) ([678fbe8](https://github.com/fsafaei/concerto/commit/678fbe82627aa4802f39eac7b4aa7ade2dd7d58d))
* **envs:** add co-carry coupling-valid rig (Rungs 0-1) per ADR-026 ([#245](https://github.com/fsafaei/concerto/issues/245)) ([b8a7734](https://github.com/fsafaei/concerto/commit/b8a77349c1ef5464bee2f10cb0ecb469b789c436))
* **envs:** add Stage1PickPlaceEnv real ManiSkill v3 env for ADR-007 Stage 1b ([#161](https://github.com/fsafaei/concerto/issues/161)) ([53f9198](https://github.com/fsafaei/concerto/commit/53f9198194c960e49b109ce342614482aa57e6d7))
* **envs:** co-insert hold-and-insert S0 — env skeleton + pre-registration ([#256](https://github.com/fsafaei/concerto/issues/256)) ([729c41d](https://github.com/fsafaei/concerto/commit/729c41d4baa3a2a874a1b58dd0b7ba1a7d1a744b))
* **envs:** co-insert S1 — peg-socket contact + MuJoCo oracle (stay verdict) ([#257](https://github.com/fsafaei/concerto/issues/257)) ([d81726b](https://github.com/fsafaei/concerto/commit/d81726bf1018ccc0c3804ca02f7ee29089ed20a8))
* **envs:** co-insert S2 — honest close at HARD_STOP (sim wall disproven; geometric tilt-wedge) ([#258](https://github.com/fsafaei/concerto/issues/258)) ([a9df96c](https://github.com/fsafaei/concerto/commit/a9df96c437ab1a4702ef3a8fbb9d0195dda10b9f))
* **envs:** vectorise the Stage-1b training cell (num_envs &gt; 1) + ADR-007 Rev 17 ([#213](https://github.com/fsafaei/concerto/issues/213)) ([e82d0ed](https://github.com/fsafaei/concerto/commit/e82d0ed5b8c62d81370795c1d6745d7be0a2c1fb))
* **evaluation:** OSCBF slack telemetry → ADR-014 Table 2 ConditionRow (closes [#142](https://github.com/fsafaei/concerto/issues/142)) ([#171](https://github.com/fsafaei/concerto/issues/171)) ([39216b8](https://github.com/fsafaei/concerto/commit/39216b82cecacd2a151e39e1a0c90744938f3933))
* **handover-place-gate0:** Gate-0 harness + Stage-0 pre-check + pre-registration (PRs A+B) ([#262](https://github.com/fsafaei/concerto/issues/262)) ([1b6f38f](https://github.com/fsafaei/concerto/commit/1b6f38f386e775166cad355cfb206089c97a2750))
* **handover-place-gate0:** run Gate-0 and commit immutable results archive ([#263](https://github.com/fsafaei/concerto/issues/263)) ([645ab7a](https://github.com/fsafaei/concerto/commit/645ab7ab66159ee661985160526ba62b3ac0d6c9))
* **observability:** wandb + chamber-analyze + rollout dumps for Phase-1 RL co-analysis (P1.05.11) ([#189](https://github.com/fsafaei/concerto/issues/189)) ([760b5f4](https://github.com/fsafaei/concerto/commit/760b5f43f86cc06acf5e9b5380d54372824b10e7))
* **partners:** EXPLORATORY partner-static override, structurally barred from gate-facing runs ([#234](https://github.com/fsafaei/concerto/issues/234)) ([b585e4b](https://github.com/fsafaei/concerto/commit/b585e4bcb199151e244b34d7b7b31a4055aedea8))
* **safety:** symmetric lambda clamp + clamp_floor_ratio config (closes [#180](https://github.com/fsafaei/concerto/issues/180); ADR-004 Rev 2026-05-20) ([#180](https://github.com/fsafaei/concerto/issues/180)) ([c6a03cf](https://github.com/fsafaei/concerto/commit/c6a03cf7b18ce74c9e59b64610d8d556943c3376))
* **training,safety:** wire CBF-QP outer filter + λ capture into ego-AHT training rollout (P1.04.5; ADR-007 §Stage 1b) ([#166](https://github.com/fsafaei/concerto/issues/166)) ([e4a37ec](https://github.com/fsafaei/concerto/commit/e4a37ec715f4c2ae22933e04468276b08e690d36))
* **training,safety:** wire maybe_brake in training rollout for deployment-time parity (P1.04.6; ADR-007 §Stage 1b Rev 8) ([#174](https://github.com/fsafaei/concerto/issues/174)) ([ae533c4](https://github.com/fsafaei/concerto/commit/ae533c4af857686e597102e08d6316be39203590))
* **training:** bump Stage-1b total_frames 100k→1M + happo.hidden_dim 64→256 (P1.05.9; second §4a closure attempt; ADR-007 Rev 14; ADR-002 Rev 2026-05-23) ([#200](https://github.com/fsafaei/concerto/issues/200)) ([117a012](https://github.com/fsafaei/concerto/commit/117a0121b535dfa71c1c1b781d1c35eb13340c9c))
* **training:** potential-based settle shaping behind shaping.settle_alpha (ADR-007 Rev 18) ([#219](https://github.com/fsafaei/concerto/issues/219)) ([87f8deb](https://github.com/fsafaei/concerto/commit/87f8deb0e842e8c1c6000b9b037755a152c8b84e))


### Bug Fixes

* **cli:** chamber-spike next-stage skips Stage-1a archives with stderr note (closes ADR-016 §Open questions) ([#155](https://github.com/fsafaei/concerto/issues/155)) ([01ff673](https://github.com/fsafaei/concerto/commit/01ff67309ab5e48d8b16cf7c5895172fe0435ffe))
* **cocarry:** Rung-2 reward remediation — stress penalty + transport PBRS (0%→67%, STOP at budget) ([#247](https://github.com/fsafaei/concerto/issues/247)) ([0bf0601](https://github.com/fsafaei/concerto/commit/0bf06018103db502813f6289588c5b2995ded6d2))
* **deps:** bump tornado 6.5.5 → 6.5.7 to clear four OSV advisories ([#242](https://github.com/fsafaei/concerto/issues/242)) ([27a40d6](https://github.com/fsafaei/concerto/commit/27a40d6f9e06f42486cda7383613fe6687aa9e83))
* enforce Stage-1b episode_length horizon (env truncation + eval-cap) — P-RC root-cause fix ([#206](https://github.com/fsafaei/concerto/issues/206)) ([01a81f0](https://github.com/fsafaei/concerto/commit/01a81f0884e79fe078693c09f3e1a3addc2229aa))
* **env:** initialise Stage-1b robot qpos to canonical ready pose + open gripper (P1.05.9; ADR-007 Rev 16; ADR-002 determinism re-check) ([#210](https://github.com/fsafaei/concerto/issues/210)) ([5edfbfb](https://github.com/fsafaei/concerto/commit/5edfbfbd9f974c5e52c590433b1a65c782a42503))
* **envs,benchmarks,tests:** widen Stage1ASStateSynthesizer to inject task + partner state; drop rubber-stamp success rule (P1.05.8; closes Surface 6 / Surface 1; ADR-002 Rev 2026-05-21; ADR-009 Rev 2026-05-21; ADR-007 Rev 12) ([#185](https://github.com/fsafaei/concerto/issues/185)) ([dd02b80](https://github.com/fsafaei/concerto/commit/dd02b800f64302f08814c72549a5df1720088c62))
* **envs,benchmarks:** Stage1PickPlaceEnv state synthesis + zero-factory ego_uid (closes P1.04 Tier-2 gap) ([#163](https://github.com/fsafaei/concerto/issues/163)) ([e1a84f7](https://github.com/fsafaei/concerto/commit/e1a84f737aa9c003b67b35a6e2b8f0e98b3228e1))
* **envs,tests:** wire Stage1OMChannelFilter into factory + Tier-1 fake state_dict key alignment (closes [#165](https://github.com/fsafaei/concerto/issues/165)) ([#167](https://github.com/fsafaei/concerto/issues/167)) ([6f98ba6](https://github.com/fsafaei/concerto/commit/6f98ba66d9a3ee65de63be5932c80cfdf53b2d1a))
* **envs:** device-aware zero-masking in Stage1OMChannelFilter ([#236](https://github.com/fsafaei/concerto/issues/236)) ([706a3f7](https://github.com/fsafaei/concerto/commit/706a3f7a07acb5d0609ebd35c7823d0dac8a7e5b))
* **envs:** settle-shaping Phi reads privileged env state, not the filtered obs ([#237](https://github.com/fsafaei/concerto/issues/237)) ([e3d516d](https://github.com/fsafaei/concerto/commit/e3d516d625134fab44266920391fe35cb56daed3))
* **envs:** Stage0StateSynthesizer at make_stage0_env outer wrap — closes [#184](https://github.com/fsafaei/concerto/issues/184) ([#192](https://github.com/fsafaei/concerto/issues/192)) ([99c7cbc](https://github.com/fsafaei/concerto/commit/99c7cbc01ad913e9918b23ab95acc751d86ed3cd))
* gate-spike blockers B1 ([#215](https://github.com/fsafaei/concerto/issues/215)) + B2 ([#214](https://github.com/fsafaei/concerto/issues/214)) — adapter dispatch order and run_id config fingerprint ([#217](https://github.com/fsafaei/concerto/issues/217)) ([b5d15da](https://github.com/fsafaei/concerto/commit/b5d15da64c2b75671fbc525413acc168f862730c))
* **safety:** Bounds field split (closes [#146](https://github.com/fsafaei/concerto/issues/146)) + JacobianEmergencyController for 7-DOF arms (ADR-004 risk-mitigation [#1](https://github.com/fsafaei/concerto/issues/1)) ([#157](https://github.com/fsafaei/concerto/issues/157)) ([571ed7f](https://github.com/fsafaei/concerto/commit/571ed7f62854eb9a9f0af55043bbc63b4d667685))
* **safety:** symmetric predicate A in Stage-1b audit-gate hooks (closes [#178](https://github.com/fsafaei/concerto/issues/178); ADR-007 §Stage 1b Rev 10) ([#179](https://github.com/fsafaei/concerto/issues/179)) ([0cb5c50](https://github.com/fsafaei/concerto/commit/0cb5c50322971d38ea572f9b4f4e48e6ba482f96))
* **tests:** isolate test_verify_git_tag_detects_mismatch from host tag.gpgsign ([#191](https://github.com/fsafaei/concerto/issues/191)) ([f9f7ec0](https://github.com/fsafaei/concerto/commit/f9f7ec0cd92f4d5af75625ff4f0eadecb4199535))
* **tests:** override scripted_heuristic action_dim for Stage-0 round-trip — closes [#194](https://github.com/fsafaei/concerto/issues/194) ([#195](https://github.com/fsafaei/concerto/issues/195)) ([383532c](https://github.com/fsafaei/concerto/commit/383532cce2803b820831413f255decb45f2ce44c))
* **tests:** per-uid action_dim in _zero_dict_action — closes [#196](https://github.com/fsafaei/concerto/issues/196) ([#197](https://github.com/fsafaei/concerto/issues/197)) ([efe2d7b](https://github.com/fsafaei/concerto/commit/efe2d7b31892089e60b299c948a8b0849d37f498))
* **tests:** tighten coverage detection + zoo-seed skip predicates + Jacobian action dim (closes [#164](https://github.com/fsafaei/concerto/issues/164)) ([#168](https://github.com/fsafaei/concerto/issues/168)) ([437b19d](https://github.com/fsafaei/concerto/commit/437b19d7db336902698e581d604505e985160bc1))
* **training:** resolve git_sha once per process for run provenance ([#238](https://github.com/fsafaei/concerto/issues/238)) ([b347866](https://github.com/fsafaei/concerto/commit/b3478663d2d2b15673b1e4476d4075aa888206ce))
* **utils:** add sapien_cuda_renderer_available() — closes [#188](https://github.com/fsafaei/concerto/issues/188) ([#193](https://github.com/fsafaei/concerto/issues/193)) ([79b487d](https://github.com/fsafaei/concerto/commit/79b487d2d2379b4bc469b357b693d0fd2cea0649))


### Documentation

* **adr:** add ADR-026 coupling-validity criterion (RFC) + index entry; reinterpret the Stage-1 AS verdict as construct-invalid ([#241](https://github.com/fsafaei/concerto/issues/241)) ([a1c16bb](https://github.com/fsafaei/concerto/commit/a1c16bba1f47389d63adb8724a1afde21599ed8e))
* **coinsert:** reconcile seat-depth figure to the 38 mm seated threshold ([#259](https://github.com/fsafaei/concerto/issues/259)) ([2b78f20](https://github.com/fsafaei/concerto/commit/2b78f2035f5d7ba8a7e44934a491bc16efdaaa8b))
* land P1.05.9 horizon-fix record — brief resolution + ADR-007 Rev 15 + ADR-002 amendment ([#208](https://github.com/fsafaei/concerto/issues/208)) ([27702da](https://github.com/fsafaei/concerto/commit/27702daaeacda8f5a956a5fa135c927d5e363d7b))
* **observability:** cookbook worked examples + config and JSON-output schema reference ([#190](https://github.com/fsafaei/concerto/issues/190)) ([8e9a93d](https://github.com/fsafaei/concerto/commit/8e9a93d7ee182c77bd731feb637c03efa9cdcaff))
* reconcile public docs with ADR-026 (coupling-validity) ([#243](https://github.com/fsafaei/concerto/issues/243)) ([97b1afc](https://github.com/fsafaei/concerto/commit/97b1afc08f580a1ffeeb38fcdf219cc0a4a0e16c))
* **repro:** no repository mutations during a live chain (operational note) ([#228](https://github.com/fsafaei/concerto/issues/228)) ([f3cd638](https://github.com/fsafaei/concerto/commit/f3cd6380ba5d146dae665628a1dc110db4b1f144))
* **spikes:** EXPLORATORY homo-static characterization — the moving partner does NOT explain the reversal (0/90, frozen rule) ([#235](https://github.com/fsafaei/concerto/issues/235)) ([24bfb5e](https://github.com/fsafaei/concerto/commit/24bfb5edecc8f6567e265f575df16441c5708abc))
* **spikes:** gamma-scan characterization — no gate-viable gamma; the discount lever is exhausted ([#218](https://github.com/fsafaei/concerto/issues/218)) ([82d1bda](https://github.com/fsafaei/concerto/commit/82d1bda43d950b401670e62d216ff0d910030ab8))
* **spikes:** gate-regime spec (founder-approved) + AS-homo path smoke at the gate regime ([#221](https://github.com/fsafaei/concerto/issues/221)) ([b560466](https://github.com/fsafaei/concerto/commit/b560466c81ab007acd79c237e55da08c58ed0ff9))
* **spikes:** gate-verdict processing — operationalization audit, consultation brief, OM assessment, follow-up drafts, register/status inputs ([#229](https://github.com/fsafaei/concerto/issues/229)) ([3b40c30](https://github.com/fsafaei/concerto/commit/3b40c301c1fc2d1fbcf87c2f089f2924f3b05220))
* **spikes:** grasp-remediation campaign log (PA1/PA2 failed, PA3 in progress) ([#209](https://github.com/fsafaei/concerto/issues/209)) ([dca03ad](https://github.com/fsafaei/concerto/commit/dca03adbbfa03b6539de8df85f07a125b71dc300))
* **spikes:** PBRS-settle characterization — the lever works at alpha=0.5 (first qualifying success in the campaign) ([#220](https://github.com/fsafaei/concerto/issues/220)) ([dde63b5](https://github.com/fsafaei/concerto/commit/dde63b5643789a50aad2bf40c9d4ea5848320255))
* **spikes:** regime-alignment characterization — consolidation explained, campaign closes with zero levers ([#212](https://github.com/fsafaei/concerto/issues/212)) ([788abcc](https://github.com/fsafaei/concerto/commit/788abcc1dff111490bc419277bfcfb6d2a5d316d))
* **spikes:** reset-fix cold-eval + 3-seed characterization (grasp emergence replicates in training; cold bar not met) ([#211](https://github.com/fsafaei/concerto/issues/211)) ([e7005ee](https://github.com/fsafaei/concerto/commit/e7005ee10f1708b864016ebf7c1c8a6bacef4ccd))
* **spikes:** Stage-1b AS gate verdict — exit 5; the gap is sign-reversed (hetero 12/100 vs homo 1/100) ([#226](https://github.com/fsafaei/concerto/issues/226)) ([4150c7f](https://github.com/fsafaei/concerto/commit/4150c7f6d17e85591dd2cd6f9b29ef42392e4369))
* **spikes:** success-static decomposition probe — C1 verdict + options + gate-spike readiness ([#216](https://github.com/fsafaei/concerto/issues/216)) ([c3efb0f](https://github.com/fsafaei/concerto/commit/c3efb0f3f4e8848819007cd70bcce8ad918cb37e))

## [0.7.0](https://github.com/fsafaei/concerto/compare/v0.6.0...v0.7.0) (2026-05-17)


### Features

* **evaluation,benchmarks:** SpikeRun sub_stage field + summarize-month3 Stage-1a routing ([#152](https://github.com/fsafaei/concerto/issues/152)) ([a89a260](https://github.com/fsafaei/concerto/commit/a89a26031863657cd8270c6abb6f866e9452243b))


### Bug Fixes

* **spikes:** summarize-month3 ignores leaderboard.json + stage1 adapter records prereg_sha ([#150](https://github.com/fsafaei/concerto/issues/150)) ([8a13260](https://github.com/fsafaei/concerto/commit/8a13260464f5d09bd3f5dc6242ded544c7a93fe4))


### Documentation

* **explanation:** capture Month-3 lock-priority report (2026-05-17 Defer state) ([#154](https://github.com/fsafaei/concerto/issues/154)) ([2bc86fe](https://github.com/fsafaei/concerto/commit/2bc86fee434ba7b58a4c02c102ac5590e6d2886b))

## [0.6.0](https://github.com/fsafaei/concerto/compare/v0.5.0...v0.6.0) (2026-05-17)


### Features

* **safety:** OSCBF returns slack telemetry + ADR-014 Table 2 schema-v2 (closes review P0-3) ([#141](https://github.com/fsafaei/concerto/issues/141)) ([3b30001](https://github.com/fsafaei/concerto/commit/3b300018e831ef09b3df84ab66a0ee636081764f))


### Bug Fixes

* **safety:** scale predicted Cartesian acceleration by dt (closes review P0-1) ([#140](https://github.com/fsafaei/concerto/issues/140)) ([c342d9f](https://github.com/fsafaei/concerto/commit/c342d9f847adb409e213e3623b0cf2cd3ba16c8f))


### Documentation

* **release:** sync CITATION.cff to v0.6.0 + add release-please extra-files config ([#148](https://github.com/fsafaei/concerto/issues/148)) ([4e8da9c](https://github.com/fsafaei/concerto/commit/4e8da9c3dd2e70586a3998ce4183bee42a8f3275))
* **safety:** mark Bounds.action_norm semantic inconsistency + xfail (closes review P1-3) ([#147](https://github.com/fsafaei/concerto/issues/147)) ([281fa2f](https://github.com/fsafaei/concerto/commit/281fa2fccf6ae8b40193e452ed9fe4c884879ba3))
* **safety:** mark lambda_safe=0.0 as Phase-0 placeholder (closes review P0-4) ([#138](https://github.com/fsafaei/concerto/issues/138)) ([9bd9397](https://github.com/fsafaei/concerto/commit/9bd9397f9b2163d9f170098d55ea6551bf5d1b3f))

## [0.5.0](https://github.com/fsafaei/concerto/compare/v0.4.0...v0.5.0) (2026-05-16)


### Features

* **ci:** verify-changelog-completeness guardrail (closes [#126](https://github.com/fsafaei/concerto/issues/126)) ([#135](https://github.com/fsafaei/concerto/issues/135)) ([255940e](https://github.com/fsafaei/concerto/commit/255940ee81907b5752be3228de4150e913aaf70c))


### Documentation

* **readme:** sync TestPyPI badge + CITATION.cff for v0.4.0 ([#137](https://github.com/fsafaei/concerto/issues/137)) ([e83e940](https://github.com/fsafaei/concerto/commit/e83e9407a91dec99c93bcc9e1dea0b737eba2e78))

## [0.4.0](https://github.com/fsafaei/concerto/compare/v0.3.1...v0.4.0) (2026-05-16)


### Features

* **packaging:** move HARL to [dependency-groups].train + lazy-import (closes [#131](https://github.com/fsafaei/concerto/issues/131)) ([#133](https://github.com/fsafaei/concerto/issues/133)) ([ac9ef22](https://github.com/fsafaei/concerto/commit/ac9ef228b29fe46313d5f9ca7ec929e6006da036))

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
