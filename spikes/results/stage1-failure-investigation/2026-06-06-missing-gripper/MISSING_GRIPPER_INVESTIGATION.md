# Missing-gripper render anomaly — the panda hand is present, actuated, and rendered

**Date.** 2026-06-06.
**Branch.** `feat/p1-05-9-p1-reward-bridge` (checkpoint `git_sha=4e9d2a6`).
**Author.** Farhad Safaei.
**Trigger.** A rollout of the current Stage-1b checkpoint (AS-hetero,
`stage1_pickplace_panda_plus_fetch_ego_aht_happo_per_agent`, run
`ee0e52a9afae6b78`, step 500000) rendered the ego `panda_wristcam` arm
*appearing* to end at the wrist flange — no hand, no fingers, no wristcam
mount. Grasp learning has plateaued (`ever_grasped = 0/5` at eval), so before
treating the plateau as a behavioural problem we had to rule out the
embodiment being physically broken: if the hand were genuinely absent from the
physics model, grasping would be impossible and that would be the blocker
itself, not a cosmetic issue.

## Headline

**The embodiment is intact. None of H1/H2/H3 holds.** The `panda_wristcam` hand
and both fingers are present in the loaded physics model, are actuated (action
dim 8, the checkpoint's action head is 8-D), have visual meshes that load from
disk and rasterise on screen, and are built identically by the training env
path and the rollout/visualisation env path. The "arm ends at the wrist flange"
observation is a **rendering-scale perception artefact**: at the fixed
third-person 640×480 camera the closed-gripper hand+wristcam assembly subtends
~10–15 px, and the env resets with the gripper fully closed (fingers pressed
together into a small nub), so the assembly reads as the arm's terminus. Zoomed
5× it is unmistakably the Panda hand, wristcam barrel, and fingers.

**Consequence for the plateau.** Because the gripper is fully functional
(action dim 8; fingers actuate `0.0 → 0.063 → 0.0 m` on command), the grasp
plateau is a *behavioural* failure of the policy, not an embodiment defect. The
operative cause is upstream of the gripper: the arm under-reaches (closest TCP→
cube distance ~12 cm across eval episodes, never within the ~2 cm grasp range,
even when the eval horizon is extended 4×). This is the same surface the
planned P1.05.9 reach→grasp remediation targets.

## Hypotheses and verdicts

| ID | Hypothesis | Verdict | Decisive evidence |
|----|-----------|---------|-------------------|
| H1 | Hand absent from the loaded physics model (wrong URDF / registration shadowing / missing assets) ⇒ action dim 7, grasping impossible | **REFUTED** | `panda_wristcam` resolves `panda_v3.urdf` with 15 links incl. `panda_hand`, `panda_leftfinger`, `panda_rightfinger`; 9 active joints (7 arm + 2 prismatic fingers); env action space `Box(8,)`; **checkpoint actor head `fc_mean (8,256)`, `log_std (8,)` → 8-D**; fingers actuate `0.0 → 0.063 → 0.0 m` |
| H2 | Hand present in physics but visual meshes fail to load (render-only defect) ⇒ action dim 8; plateau has a different cause | **REFUTED** (core claim) | `panda_hand`/`leftfinger`/`rightfinger` each report `render_shapes = 1`; `hand.glb` (590 608 B) + `finger.glb` (50 032 B) exist and are non-empty; **closed-vs-open render diff: 2 433 px change localised to the hand bbox (309,0)–(487,141)** — the finger meshes rasterise. (Its *secondary* claim — action dim 8, plateau is behavioural — is correct.) |
| H3 | Divergence between the training env build path and the rollout/recorder build path | **REFUTED** | Both paths route through the single factory `make_stage1_pickplace_env`; the adapter `_stage1b_env_factory` calls it directly (`stage1_as.py:526`); the rollout/visualiser calls it with only `render_mode`/`render_backend` differing. A↔B embodiment diff is identical for every field (n_links, has_hand, n_finger, active_joints, urdf). The `RolloutRecorder` builds **no** env — it is a passive consumer of caller-supplied obs/action/frame |

**Reported conclusion: NONE of H1/H2/H3.** The embodiment and render pipeline are
correct; the observation is a perception artefact at native render scale. No
additional evidence is required to discriminate — the physics articulation, the
checkpoint action head, the on-disk assets, the rasterised pixels, and the
build-path equivalence are all directly measured and mutually consistent.

## Method

Throwaway probes (under `/tmp`, not committed) built the AS-hetero env two ways
and dumped, per agent uid: resolved URDF path, link names, active joint names,
render-shape counts for hand/finger/camera links, and the env action space;
loaded the checkpoint actor and printed its action-head output dim; drove the
gripper open/closed in isolation reading finger qpos; verified the on-disk
URDF + mesh assets; and ran a same-pose gripper closed-vs-open render diff to
localise where (if anywhere) the finger meshes draw. Stdout is archived here as
`probe_embodiment.log`, `asset_integrity.log`, and `render_diff.log`; frames as
`probe_render_frame.png` and `render_gripper_{closed,open,diff}.png` +
`render_gripper_zoom_closed_vs_open.png`.

Host/versions: `mani_skill 3.0.1`, `sapien 3.0.3`, `torch 2.11.0+cu128`.

## Evidence

### E1 — Physics articulation (refutes H1)

```
uid='panda_wristcam'  (PandaWristCam)
  urdf_path     = .../mani_skill/assets/robots/panda/panda_v3.urdf
  n_links       = 15
  links         = [... 'panda_link7','panda_link8','panda_hand','panda_hand_tcp',
                   'panda_leftfinger','panda_rightfinger','camera_base_link','camera_link']
  active_joints = ['panda_joint1'..'panda_joint7','panda_finger_joint1','panda_finger_joint2']  (9)
  single_action_space = Box(-1.0, 1.0, (8,), float32)
```

The control stack is `CombinedController(arm=PDJointPosController,
gripper=PDJointPosMimicController)`; the 8-D action is 7 arm joint deltas + 1
mimic-gripper delta under `pd_joint_delta_pos`.

### E2 — Checkpoint action head (refutes H1)

```
act.action_out.fc_mean.weight  (8, 256)
act.action_out.fc_mean.bias    (8,)
act.action_out.log_std         (8,)
trainer.act(...) ego action dim = (8,)
```

A 7-D head against an 8-D env (or the reverse) would be the decisive
H1 signature; the head is 8-D, matching the env. The gripper DOF is in the
controlled action vector.

### E3 — Gripper actuates (refutes H1)

```
reset width=(0.0, [0.0, 0.0])
after 30x OPEN  +1: width=(0.06295, [0.03155, 0.0314])
after 30x CLOSE -1: width=(0.0,     [0.0,    0.0])
```

The env **resets with the gripper fully closed** (width 0.0) — relevant to the
perception artefact: the fingers start as a closed nub.

### E4 — Asset integrity (refutes H1/H2)

```
panda_v3.urdf                                              13 483 B
panda_hand       franka_description/meshes/visual/hand.glb    590 608 B  exists=True
panda_leftfinger franka_description/meshes/visual/finger.glb   50 032 B  exists=True
panda_rightfinger franka_description/meshes/visual/finger.glb  50 032 B  exists=True
```

### E5 — Visual meshes rasterise (refutes H2) — **single most decisive line**

```
render_shapes: panda_hand=1, panda_leftfinger=1, panda_rightfinger=1, camera_link=1
closed-vs-open render diff: changed pixels (>20 sum-RGB) = 2433
                            diff bounding box (x0,y0,x1,y1) = (309, 0, 487, 141)
```

Holding the arm pose fixed and toggling only the gripper changes 2 433 pixels,
all inside the hand bounding box. Pixels that change with finger motion are, by
definition, finger pixels being drawn. `render_gripper_zoom_closed_vs_open.png`
shows the hand, wristcam barrel, and fingers directly.

### E6 — Single build path, identical embodiment (refutes H3)

```
[A vs B diff] embodiment equality across build paths
  uid='panda_wristcam': n_links=True has_hand=True n_finger=True active_joints=True urdf=True
  uid='fetch':          n_links=True has_hand=True n_finger=True active_joints=True urdf=True
```

`_stage1b_env_factory` (training/eval dispatch) → `make_stage1_pickplace_env`
(`stage1_as.py:526`); the rollout visualiser → the same factory with only
`render_mode="rgb_array"`/`render_backend="gpu"` differing. The
`patch_sapien_urdf_no_visual_material()` material-strip shim runs **only** when
`render_backend == "none"` (not the GPU render path) and strips material/colour,
not geometry. `load_agent_with_bare_uids` only renames `agents_dict` keys
(strips the `-i` suffix) — it never touches links, visuals, or meshes.

## Why the arm looks like it ends at the flange

Three compounding factors, none a defect:

1. **Closed-at-reset gripper.** The fingers reset to width 0.0 — pressed
   together into a small nub rather than the splayed "claw" silhouette the eye
   expects.
2. **Render scale.** At the fixed `look_at(eye=[0.7,0.7,0.6],
   target=[0,0,0.1])` 640×480 third-person camera, the hand+wristcam assembly
   is ~10–15 px; GIF quantisation smears it further.
3. **Pose in the inspected episode.** In the rollout episode that prompted the
   report the arm stays raised and away from the table (closest approach is at
   reset, then it drifts up), so the hand sits high and small near the frame
   edge, where the wristcam barrel reads as the arm's end.

## Recommended remediation — diagnosis only; no code changed in this pass

This investigation modified no library code, configs, schemas, or prereg YAMLs.

1. **No embodiment fix is warranted** — there is no embodiment or render-pipeline
   defect. The "missing gripper" is closed.

2. **Observability ergonomics (cosmetic, optional).** To stop future rollout
   inspections from misreading the hand as absent, propose a follow-up that
   either (a) raises the render-camera resolution, or (b) adds a second,
   closer "wrist inspection" `CameraConfig` framed on the TCP, for
   visualisation only. This grounds in **ADR-017 §Decisions** (the rollout
   recorder / MP4 + per-step JSONL observability surface) and **ADR-007 §Stage 1b**
   (the `_default_human_render_camera_configs` render camera is explicitly
   render-only and feeds no observation channel or gate contract, so changing it
   does not touch determinism or the comparison protocol). Render-only ⇒ no new
   prereg tag required.

3. **The real blocker is the approach, and it is already in scope.** Because the
   gripper works, the plateau is the under-reaching behaviour (arm asymptotes
   ~12–19 cm from the cube). That is exactly the surface the **P1.05.9
   reach→grasp reward bridge** addresses — the experimental, disabled-by-default
   `reward_bridge_*` path already in `make_stage1_pickplace_env`
   (remediation proposal 2026-06-05 §P1, target hypothesis H2a; **ADR-007
   §Discipline Rev-15 PENDING**). This investigation **removes "broken
   embodiment" from that remediation's hypothesis set**: any reach→grasp bridge
   experiment can proceed on the premise that grasping is physically possible
   and the gripper DOF is in the controlled action vector. Note the run that
   produced this checkpoint (`ee0e52a9afae6b78`, bridge weight 0.15, 500 k
   frames) still showed `ever_grasped = 0/5` — i.e. the bridge as configured did
   not lift grasping, which is a result for the P1.05.9 slice to weigh, not an
   embodiment question.

## Relation to the P1.05.9 remediation slice

P1.05.9 (third §4a firing, AS-hetero grasp plateau) is mid-remediation: the
reach→grasp reward bridge is implemented but disabled-by-default pending the
senior-advisor consultation (ADR-007 §Discipline Rev-15). This investigation is
an **embodiment-integrity gate** in front of that slice. Verdict: the
embodiment is sound; the plateau is behavioural; the bridge experiment may
proceed without an embodiment caveat. The cosmetic observability item (§2
above) is independent and non-blocking.
```
SHA-256 manifest: SHA256SUMS.txt (this directory; prior investigation
directories are immutable and untouched).
```
