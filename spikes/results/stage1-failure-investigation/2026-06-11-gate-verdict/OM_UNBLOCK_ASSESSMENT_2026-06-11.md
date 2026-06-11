# OM unblock assessment — what #177 actually takes at the vectorised regime

**Date.** 2026-06-11.
**Author.** Farhad Safaei.
**Status.** Phase-3 deliverable of the gate-verdict processing. **Assessment only — nothing
is implemented here**; the fix plan is a draft for founder approval in its own slice. The
trivial-fix branch ("≤ ~1 h, unambiguously rig-class → say so and stop") does **not** apply:
the unblock is multi-day work, and this probe surfaced two additional defects #177 predates.

## §1 — What exactly is broken (verified this session)

1. **The known blocker (#177, Rev 9 deferral):** `EgoPPOTrainer.from_config` consumes only a
   flat `obs["agent"][ego]["state"]` Box; the OM wrapper chain deliberately omits it
   (`Stage1ASStateSynthesizer` is pass-through for `is_om_condition=True`), so trainer
   construction raises `KeyError: 'state'` for both OM conditions. Rev 9 explicitly bars the
   shortcut of synthesising state for OM — it would leak masked modalities into the
   vision-only policy and corrupt the OM-homo science. A vision-head trainer is the faithful
   fix.
2. **NEW — U1, vectorised OM env is broken before the trainer is even reached:**
   `Stage1OMChannelFilter.observation` zero-masks via `np.zeros_like(val)`
   (`stage1_obs_filter.py:681`), which raises `TypeError: can't convert cuda:0 device type
   tensor to numpy` on GPU-sim tensors. The OM-vision-only env cannot `reset()` at
   `num_envs > 1` today (reproduced at N=64/128/256). Fix: device-aware masking
   (`torch.zeros_like` for tensors, `np.zeros_like` for arrays). Small, unambiguous
   rig-class.
3. **NEW — U2, the Rev 18 settle shaping is condition-asymmetric under OM-vision-only as
   wired:** `Stage1SettleShapingWrapper` computes Φ's placement gate from
   `obs["extra"]["cube_pose"]`/`goal_pos`, and `run_training` wraps the *outer* (filtered)
   env — but the vision-only keep-set **zero-masks `cube_pose`**, so Φ's gate would read a
   zeroed cube and the shaping would differ between the two OM conditions, violating the
   Rev 18 condition-symmetry mandate. Fix: the shaping wrapper reads privileged env state
   (the cube/goal poses directly, as the reward function itself already does) rather than
   the filtered obs — training-time reward computation is privileged by construction.
   Small; symmetry-critical; needs a Tier-1 pin.

## §2 — The cost reality: "the proven regime at ~5.5 h" does NOT transfer to OM (measured)

The AS gate's 11.1k steps/s is a state-mode number. OM training renders **three RGB-D
cameras per step** (`panda_wristcam-0-hand_camera`, `fetch-1-fetch_head`,
`fetch-1-fetch_hand`; 128×128). Measured env-side stepping (random actions, no trainer,
RTX 2080 SUPER):

| N | env steps/s | device memory over baseline |
|---|---|---|
| 64 | 570 | 1.85 GB |
| 128 | 708 | 2.05 GB |
| 256 | 812 | 2.66 GB |

Rendering is the bottleneck and throughput **plateaus ≈ 800 steps/s regardless of N** —
parallelism stops paying. Env-side floor for one 20M-frame cell: **≈ 7 h**, before the CNN
forward/backward and image rollout-buffer costs (realistically 2–4× on top, image buffers at
`rollout_length × N × 3 cameras × 128×128 RGB-D` are the VRAM driver). A 10-cell OM gate at
the AS regime verbatim is therefore **O(100+ hours) local**, not 5.5. The consultation
brief's §4 sentence about inheriting "a proven regime at ≈ 5.5 h" must be corrected to "a
proven *training* regime whose wall-clock does not transfer; OM costs are rendering-bound"
before the brief is sent.

**Cost levers (pre-statement questions, decided by no one here):** (i) camera set — the
conditions keep `obs["sensor_data"]` wholesale; trimming to the ego wristcam alone would
~3× the env throughput and is plausibly keep-set-class (the filter's own documented note:
the prereg pins *modality families*, not per-field keep-sets — ADR-007 §Discipline), but it
shapes what "vision" means and is therefore a founder/ADR-note call; (ii) an OM-specific
frame budget (symmetric across both OM conditions; needs its own evidence that vision
policies consolidate at it); (iii) A100 rental — the question REMEDIATION_LOG §8.4 closed
for AS **reopens for OM** (rendering throughput scales with the bigger card).

## §3 — The smallest faithful fix plan (DRAFT; not implemented)

Scope = #177's design, updated for the vectorised regime, plus U1/U2:

1. **U1** — device-aware zero-masking in `Stage1OMChannelFilter` (+ Tier-1 pin with CUDA-less
   fake; Tier-2 vectorised OM reset probe). ~hour-scale.
2. **U2** — `Stage1SettleShapingWrapper` placement gate from privileged env state (+ Tier-1
   symmetry pin: identical Φ under both OM keep-sets). ~hour-scale.
3. **Vision head (the #177 core):** CNN encoder (shared trunk) + per-modality heads (RGB-D;
   optional FT 1-D; optional proprio 1-D), concatenated latent into the existing HAPPO
   actor/critic; dispatch keyed off the env's observation space (state Box present → current
   path; `sensor_data` present → vision path), per #177's env-driven contract. Image rollout
   buffers sized for the vectorised regime (uint8 storage, on-GPU normalisation, rollout
   length × N budget against 8 GB). Tier-1 fakes for the dispatch + buffer shapes; Tier-2
   construction/short-rollout smokes for both OM conditions; the two skipped OM Tier-2
   dispatch smokes re-enabled in the same PR (the #177 acceptance contract).
4. **Throughput/VRAM benchmark** at candidate (N, camera-set, rollout) before any
   pre-statement, exactly as the regime-alignment slice did for state mode.

**Classification:** items 1–3 are rig/trainer-capability work — Rev 9 already classified the
vision head as the faithful fix; no prereg YAML/tag rotation; `SCHEMA_VERSION` untouched.
The only item that grazes the axis *definition* is the camera keep-set lever in §2(i) —
keep-set-class per the filter's documented note, but a founder call with an ADR-007 note if
exercised. **Effort estimate: 2–4 founder-days** (the #177 estimate of 800–1200 LOC stands;
the vectorised-regime buffer work replaces its single-env stability concerns; U1/U2 are
small).

## §4 — Bottom line for the consultation's ask (ii)

OM's evidential value is high (no homo-operationalization confound) but its cost is **not**
5.5 h — it is multi-day engineering plus a rendering-bound gate whose wall-clock depends on
camera-set and hardware decisions not yet made. The exploratory homo-AHT cell (≈ 1.7 h,
zero new engineering) is now strictly the cheaper first probe of the §2(a)-vs-(b) question;
OM remains the only path to a confound-free axis measurement. Both can proceed without
contention (the homo cell needs no new code).

## Cross-references

Issue #177 (scope quoted; size estimate); ADR-007 §Stage 1b Rev 9 (the deferral + the
state-synthesis bar) + Rev 17/18 (regime + shaping symmetry mandates); ADR-007 §Discipline
(keep-set vs prereg-pinned); `stage1_obs_filter.py:681` (U1); `stage1_shaping.py` Φ gate
(U2); `spikes/preregistration/OM.yaml` @ `prereg-stage1-OM-2026-05-15`;
`CONSULTATION_BRIEF_GATE_VERDICT_2026-06-11.md` §4–§5 (this assessment feeds asks ii/iii);
REMEDIATION_LOG §8.4 (the A100 question, reopened for OM).
