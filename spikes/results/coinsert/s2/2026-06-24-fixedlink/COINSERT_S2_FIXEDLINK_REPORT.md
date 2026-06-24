# Co-insert S2 — fixed-link articulation attach: the create_drive wall is an artifact (2026-06-24)

**Slice.** The founder-authorised LAST bounded SAPIEN attempt after the `create_drive`
"constraint-fidelity wall": hold the socket as a **fixed link on the holder articulation**
(a fixed joint off `panda_hand`) instead of a `create_drive` maximal-coordinate inter-body
weld. Phase-2, non-gating. **Pre-S3-freeze; DO NOT MERGE.**

**Verdict.** **NOT a SAPIEN constraint-fidelity wall** — the earlier wall was a `create_drive`
maximal-coordinate artifact. The fixed-link attach removes the over-constraint and braces the
axial reaction without instability. The matched pair does **not yet reach the ≥0.9 seat**, but
the blocker is now **controller / holder-pose tuning** (peg engagement), **not** the sim. The
escalation condition ("fixed-link also cannot brace / hold alignment without instability") is
**not** met → do **not** migrate-or-close.

**Evidence** (committed `coinsert_s2_fixedlink_probe.json`, this dir; reproduce:
`uv sync --all-extras --group dev --group oracle && uv run python
scripts/repro/_coinsert_s2_fixedlink_probe.py`; seeds {0,1,2}, deterministic).

## Findings

| probe | create_drive (prior) | fixed-link (this) |
|---|---|---|
| zero-action holder-wrist preload | 573–1306 N (over-constraint) | **0.2 N** |
| matched assembly sink (drag) | ~210 mm (lateral hold) | **−1.4 mm** (braces) |
| socket orientation under load | flops / wedge / preload | align ~0.7°, stable |
| single-inserter success (free socket) | ≈ 0 | **0** (necessity preserved) |
| matched seat rate (≥38 mm, static, settled) | 0 (wall) | 0 (peg not engaging) |

1. **The over-constraint was a `create_drive` artifact.** The 573–1306 N zero-action preload
   that made the below-hold unusable collapses to **~0 N** with the fixed link — the
   maximal-coordinate inter-body drive was the fragile component, exactly as diagnosed.
2. **The below-hold fixed link BRACES.** The matched assembly does not drag (sink −1.4 mm vs
   ~210 mm for the lateral / create_drive rig); the holder joint impedance bears the axial
   insertion reaction in compression, with stable orientation and no solver instability.
3. **Two-robot necessity preserved.** The single-inserter control (free, unheld socket actor,
   no holder) still gives success ≈ 0.

## Remaining (control tuning, not a wall)

The matched pair does not seat: the peg hovers at the approach standoff and does not commit to
the press, because in the reaching-under below-hold pose the holder's **lateral** joint-impedance
is weak, so the socket drifts laterally (~11 mm) and the base inserter never sees the alignment
it needs to press. This is controller / holder-pose-conditioning tuning — pick a
better-conditioned below-hold pose where the holder holds the socket on-axis stiffly (or stiffen
the holder lateral hold / tune the base-inserter approach→press commit). It is **not** a SAPIEN
constraint-fidelity limit (the attach braces and holds alignment cleanly).

## Implementation

The socket is injected into the holder URDF as a rigid child link of `panda_hand` (a fixed
joint; `_augmented_socket_holder_urdf` + a save/set/restore swap of `PandaPartner.urdf_path`
around the agent load, confined to the matched co-insert rig). No `create_drive` in the matched
rig; the S1 fidelity probe and the single-inserter free-socket actor are unchanged. Public
`@register_agent` + URDF path only — no `mani_skill` patch (ADR-001 §Risks / P2).
