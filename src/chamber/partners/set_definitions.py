# SPDX-License-Identifier: Apache-2.0
"""The v1 partner sets (ADR-009 §Decision as amended 2026-07-05; ADR-027 §Versioning).

Registers the scripted-stratum partner sets for the CB-04-admitted tasks
(``cocarry@1``, ``handover_place@1``) and the confirmed Tier-1 control
(``stage1_pickplace_as@1`` — controls need partners too, so partner-blind
-equivalence claims stay testable). Frozen learned members join in the
baseline-training campaign as ``version+1`` per ADR-027 §Versioning; old
result bundles keep referencing the version they ran against.

Every member's exact parameters were drawn from its committed box by
:func:`chamber.partners.sets.derive_member_params` under the
maintainer-held root seed (:data:`chamber.partners.sets.PRIVATE_PARAMS_ENV`);
public members commit the drawn literals next to their digest, private
members commit the box + digest only (published hashes, withheld
parameters). The split assignments below are **outputs of the
deterministic rule** (:func:`chamber.partners.sets.compute_split` — order
by ``partner_id``, first ``⌈0.7·N⌉`` public, no hand-picking), re-verified
at registration and in the unit suite. Note the two by-construction
transparencies this discipline cannot hide (recorded, not fudged): a
member with a degenerate box (``lo == hi``, the prereg-pinned handover
presenters) reveals its value through the committed box itself, and any
fingerprint deliberately characterises behaviour (ADR-009 as amended:
readers tell members apart without policy access).

Import side effect: importing this module populates the set registry
(:mod:`chamber.partners` does so on package import, mirroring
:mod:`chamber.tasks` / :mod:`chamber.tasks.ladder`).
"""

from __future__ import annotations

from chamber.partners.sets import (
    ParamRange,
    PartnerMemberSpec,
    PartnerSetSpec,
    register_partner_set,
)

#: Committed probe-suite cluster seeds shared by every v1 set (ADR-002 P6;
#: ADR-009 as amended — the fingerprint/floor probe is ~20 episodes:
#: 4 seeds x 5 episodes against the reference ego).
PROBE_SEEDS_V1: tuple[int, ...] = (7000, 7001, 7002, 7003)

#: Episodes per probe seed (ADR-009 as amended).
PROBE_EPISODES_PER_SEED_V1: int = 5


def _f(lo: float, hi: float) -> ParamRange:
    return ParamRange(lo=lo, hi=hi, kind="float")


def _i(lo: int, hi: int) -> ParamRange:
    return ParamRange(lo=lo, hi=hi, kind="int")


def _cocarry_nominal_box() -> dict[str, ParamRange]:
    """The nominal co-carry gain box, centred on the ADR-026 §Decision 1 defaults."""
    return {
        "kp": _f(2.3, 2.7),
        "ki": _f(0.5, 0.7),
        "step_max": _f(0.028, 0.032),
        "damping": _f(0.035, 0.045),
        "lag_steps": _i(0, 0),
    }


def _cocarry_blend_box() -> dict[str, ParamRange]:
    """The off-grid blend box (all axes jointly varied; ADR-009 as amended)."""
    return {
        "kp": _f(1.3, 3.8),
        "ki": _f(0.4, 0.8),
        "step_max": _f(0.020, 0.040),
        "damping": _f(0.02, 0.08),
        "lag_steps": _i(0, 3),
    }


@register_partner_set
def cocarry_partners_v1() -> PartnerSetSpec:
    """Co-carry set v1 — the parameterized scripted stratum (ADR-009 as amended; ADR-026).

    Eleven members built from the matched impedance controller
    (:class:`chamber.partners.cocarry_impedance.CoCarryImpedancePartner`),
    spanning stiffness (low/nominal/high ``kp``), damping (low/high),
    response timing (``step_max`` slow/fast), the bounded-lag member
    ("sluggish but competent": delayed but convergent tracking,
    ``lag_steps`` 3-6), and three all-axis blends. Every box corner was
    verified competent-but-non-accommodating on the matched-pair rig
    before commitment (success 4/4, wrist stress ≤ 118 N < f_max
    130.57, the ADR-027 canonical instrument); the quarantined
    ``static_override`` wall stays out (the co-insert Gate-0 lesson —
    enforced since CB-01 by ``tests/unit/test_exploratory_partner_static.py``).
    """

    def _member(
        name: str,
        box: dict[str, ParamRange],
        split: str,
        params: dict[str, str] | None,
        digest: str,
    ) -> PartnerMemberSpec:
        return PartnerMemberSpec(
            member_name=name,
            registry_class="cocarry_impedance",
            role="partner_arm",
            split=split,  # type: ignore[arg-type]  # validated Literal by pydantic
            param_box=box,
            params=params,
            params_sha256=digest,
        )

    nominal = _cocarry_nominal_box()
    blend = _cocarry_blend_box()
    return PartnerSetSpec(
        set_id="cocarry_partners",
        version=1,
        task_id="cocarry",
        task_version=1,
        floor=0.75,
        floor_probe="fingerprint",
        probe_seeds=list(PROBE_SEEDS_V1),
        probe_episodes_per_seed=PROBE_EPISODES_PER_SEED_V1,
        notes=(
            "Scripted stratum of the cocarry@1 set (ADR-009 as amended "
            "2026-07-05). Reference ego for the probe/floor: the default-gain "
            "matched impedance controller (ADR-026 §Decision 1). Frozen "
            "learned members join as version 2 (ADR-027 §Versioning)."
        ),
        members=[
            _member(
                "imp_nominal",
                nominal,
                "public",
                {
                    "damping": "0.0374",
                    "ki": "0.5431",
                    "kp": "2.3969",
                    "lag_steps": "0",
                    "step_max": "0.0306",
                },
                "92eb953311e1b0a0d7904f41717daa0d58d08bfee1a91b9ffa132939faf83fc3",
            ),
            _member(
                "imp_stiff_low",
                {**nominal, "kp": _f(1.1, 1.6)},
                "public",
                {
                    "damping": "0.0442",
                    "ki": "0.6616",
                    "kp": "1.1991",
                    "lag_steps": "0",
                    "step_max": "0.0286",
                },
                "a5d61ee76a1a7a8a7f9241d25fca8db825215da0babb8fcda096e2bec36073a0",
            ),
            _member(
                "imp_stiff_high",
                {**nominal, "kp": _f(3.5, 4.3)},
                "public",
                {
                    "damping": "0.0364",
                    "ki": "0.6856",
                    "kp": "3.6431",
                    "lag_steps": "0",
                    "step_max": "0.029",
                },
                "212f054214340e8e69849cf63594c71bf92fdf026cddaf37c8e4f67e3999e8ff",
            ),
            _member(
                "imp_damp_low",
                {**nominal, "damping": _f(0.015, 0.025)},
                "public",
                {
                    "damping": "0.0165",
                    "ki": "0.5409",
                    "kp": "2.6044",
                    "lag_steps": "0",
                    "step_max": "0.0308",
                },
                "25c318a230e97df9c9168d0e8f98dc9c3227e4cdada4cb45dff7d09f31f64633",
            ),
            _member(
                "imp_damp_high",
                {**nominal, "damping": _f(0.075, 0.095)},
                "public",
                {
                    "damping": "0.0817",
                    "ki": "0.5119",
                    "kp": "2.3582",
                    "lag_steps": "0",
                    "step_max": "0.0302",
                },
                "509cec1613d436a03d9538961436f9c9653f74bff53abb625207043f8c4c9ad4",
            ),
            _member(
                "imp_slow",
                {**nominal, "step_max": _f(0.016, 0.021)},
                "private",
                None,
                "f769f5f281c10dcb6b7bbec47afd9be011fd615fa3f8176edbe0f33ab2403fcd",
            ),
            _member(
                "imp_fast",
                {**nominal, "step_max": _f(0.040, 0.047)},
                "private",
                None,
                "0fe7eb258c8ed441e0f3e302684d2aa0f0de539cb921bc931401b3846ae0fed8",
            ),
            _member(
                "imp_lag_bounded",
                {**nominal, "lag_steps": _i(3, 6)},
                "public",
                {
                    "damping": "0.0353",
                    "ki": "0.6211",
                    "kp": "2.5576",
                    "lag_steps": "5",
                    "step_max": "0.0294",
                },
                "fc385b64d1b4fc869b62c8d0f19b5959693ad7b634337667d25a8716d546b74f",
            ),
            _member(
                "imp_blend_a",
                blend,
                "private",
                None,
                "66ec45f3bc06acf1ca79f5f3804aff757920c3db8c02a0cdd59675cad241c1bf",
            ),
            _member(
                "imp_blend_b",
                blend,
                "public",
                {
                    "damping": "0.0232",
                    "ki": "0.6453",
                    "kp": "1.6573",
                    "lag_steps": "1",
                    "step_max": "0.0398",
                },
                "b1a0fa0aa3ee2d96af0073a70ed040fa6d90efffaeb3a956396cd5704912ddb2",
            ),
            _member(
                "imp_blend_c",
                blend,
                "public",
                {
                    "damping": "0.0281",
                    "ki": "0.7293",
                    "kp": "1.4134",
                    "lag_steps": "2",
                    "step_max": "0.0391",
                },
                "cc417fc13ef686b0d64ab6feca899ccf5f19c57f0ff874ae4a57dbaf7f6fdcf0",
            ),
        ],
    )


#: The empty-parameter digest every learned member commits
#: (:func:`chamber.partners.sets.params_sha256` of ``{}`` — the learned
#: stratum's "parameters" are its weights; custody rides on
#: ``checkpoint_sha256`` instead).
_EMPTY_PARAMS_SHA256: str = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"

#: The five jointly-trained partner-side members admitted at v2
#: (ADR-011 §Decision as amended; ADR-027 §Versioning). Each entry is
#: ``(training seed, selected checkpoint step, payload URI, payload
#: SHA-256)`` — the per-seed SELECTED pair checkpoints from the
#: committed campaign checkpoint-selection artifacts
#: (``spikes/results/benchmark/cocarry-v1/selection/b-joint-seed*.json``;
#: the ADR-027 rule on the held-out validation partner, pair mode).
#: The partner-side actor (``actor_partner``) plays the seat; the same
#: payload's SHA-256 is the "ego hash" in the provenance line because
#: the pair checkpoint IS the joint artifact containing the ego it was
#: trained with.
_JOINT_V2_MEMBERS: tuple[tuple[int, int, str, str], ...] = (
    (
        0,
        150000,
        "local://artifacts/4ace772a2efe7dd3_step150000.pt",
        "99a4d53760f8a191cbc96a9dd43721d4d314984460a21a782395617159d0fdb4",
    ),
    (
        1,
        200000,
        "local://artifacts/24e5f7483ad7b86e_step200000.pt",
        "5ec0ba22da82db62a8e3dacac864ce160a29ae92677e32b9ecdf6f9e91d4a019",
    ),
    (
        2,
        50000,
        "local://artifacts/e2f99cc34a4c5356_step50000.pt",
        "3289d6de61465d9bf8c9f133a4dc4319d86eaf89343b9a1b8a2f22a8b09f4ace",
    ),
    (
        3,
        150000,
        "local://artifacts/461dbbcae360f85e_step150000.pt",
        "74bd1db7e5ba31f9094164566bb0f2918852cf6d67194d08565ec53fb2817658",
    ),
    (
        4,
        150000,
        "local://artifacts/e9040cbc3c3b2456_step150000.pt",
        "3107b1bde5201d0350221f527839cda7ee6ac5698c20df6f80eb80453fb5ae18",
    ),
)


@register_partner_set
def cocarry_partners_v2() -> PartnerSetSpec:
    """Co-carry set v2 — the scripted stratum + the jointly-trained stratum (ADR-027 §Versioning).

    The eleven v1 members (identities byte-stable — the parameter
    substream is keyed on ``set_id`` + ``member_name``, not version)
    plus five :class:`chamber.partners.frozen_cocarry_joint.FrozenCoCarryJointPartner`
    members: the partner-side actors of the campaign's per-seed
    SELECTED jointly-trained MAPPO pair checkpoints (ADR-011 §Decision
    as amended — provenance "trained jointly with <ego hash>" rides on
    every learned card). Old result bundles keep referencing
    ``cocarry_partners@v1`` (ADR-027 §Versioning).

    The public/private split is re-derived over the full 16-member
    roster by the deterministic rule (ADR-009 as amended: order by
    ``partner_id``, first ``⌈0.7·16⌉ = 12`` public). One label flips
    vs v1 — ``imp_lag_bounded`` becomes private. Recorded, not hidden:
    its exact v1 literals are already public knowledge (committed at
    v1 and unchanged), so the v2 withholding is formal only; the rule
    is applied verbatim rather than hand-adjusted around the corner
    case (no hand-picking is the stronger property).
    """
    v2_split: dict[str, str] = {"imp_lag_bounded": "private"}
    members: list[PartnerMemberSpec] = []
    for member in cocarry_partners_v1().members:
        split = v2_split.get(member.member_name, member.split)
        if split == member.split:
            members.append(member)
            continue
        payload = member.model_dump()
        payload["split"] = split
        payload["params"] = None if split == "private" else payload["params"]
        members.append(PartnerMemberSpec.model_validate(payload))
    for seed, step, uri, sha in _JOINT_V2_MEMBERS:
        members.append(
            PartnerMemberSpec(
                member_name=f"joint_s{seed}",
                registry_class="frozen_cocarry_joint",
                role="partner_arm",
                split="public",
                seed=seed,
                checkpoint_step=step,
                param_box={},
                params={},
                params_sha256=_EMPTY_PARAMS_SHA256,
                checkpoint_uri=uri,
                checkpoint_sha256=sha,
                provenance=(
                    f"trained jointly with the ego actor of pair checkpoint "
                    f"sha256 {sha} (training seed {seed}, selected step {step} "
                    "under the ADR-027 checkpoint-selection rule)"
                ),
            )
        )
    return PartnerSetSpec(
        set_id="cocarry_partners",
        version=2,
        task_id="cocarry",
        task_version=1,
        floor=0.75,
        floor_probe="fingerprint",
        probe_seeds=list(PROBE_SEEDS_V1),
        probe_episodes_per_seed=PROBE_EPISODES_PER_SEED_V1,
        notes=(
            "v1's scripted stratum plus the campaign's jointly-trained "
            "partner-side stratum (ADR-027 §Versioning; ADR-011 §Decision as "
            "amended). Learned members carry real checkpoint URIs + committed "
            "payload SHA-256 custody; their cross-play competence with the "
            "scripted reference ego is measured by the fingerprint/floor "
            "probe, not assumed. Split re-derived over 16 members; the "
            "imp_lag_bounded public→private flip is formal only (v1 literals "
            "remain committed history)."
        ),
        members=members,
    )


@register_partner_set
def stage1_pickplace_as_partners_v1() -> PartnerSetSpec:
    """Pick-place control set v1 — minimal 3-member scripted set (ADR-009 as amended).

    ``stage1_pickplace_as@1`` is a confirmed Tier-1 CONTROL (the CB-04
    admission report demotes it: partner-blind equivalence, gap CI
    [0, 0]). Controls need partners too — this set is what makes the
    partner-blind-equivalence claim *testable* rather than asserted.
    Three :class:`chamber.partners.heuristic.ScriptedHeuristicPartner`
    variants over the planar target; N=3 puts all members public under
    the ``⌈0.7·N⌉`` rule.
    """

    def _member(
        name: str,
        box: dict[str, ParamRange],
        params: dict[str, str],
        digest: str,
    ) -> PartnerMemberSpec:
        return PartnerMemberSpec(
            member_name=name,
            registry_class="scripted_heuristic",
            role="partner",
            split="public",
            param_box=box,
            params=params,
            params_sha256=digest,
        )

    return PartnerSetSpec(
        set_id="stage1_pickplace_as_partners",
        version=1,
        task_id="stage1_pickplace_as",
        task_version=1,
        floor=0.90,
        floor_probe="fingerprint",
        probe_seeds=list(PROBE_SEEDS_V1),
        probe_episodes_per_seed=PROBE_EPISODES_PER_SEED_V1,
        notes=(
            "Minimal scripted set for the Tier-1 CONTROL (ADR-027 §Tier "
            "ladder; CB-04 demotion). Reference ego for the probe/floor: the "
            "REF-SCRIPT scripted oracle (ADR-011 as amended). The control's "
            "point is that success is partner-insensitive — the floor "
            "documents that fact per member."
        ),
        members=[
            _member(
                "heuristic_center",
                {"target_x": _f(-0.01, 0.01), "target_y": _f(-0.01, 0.01)},
                {"target_x": "-0.006", "target_y": "-0.0065"},
                "963dcf2d674c69aae31b4417422e62465c21bd7ac99ab592706ecf8771e9c88e",
            ),
            _member(
                "heuristic_quadrant_pos",
                {"target_x": _f(0.03, 0.06), "target_y": _f(0.03, 0.06)},
                {"target_x": "0.0455", "target_y": "0.0412"},
                "951fdfbeb2caab78b14f33f316cb75d2af92abe3d132114d3421b02ba6488b62",
            ),
            _member(
                "heuristic_quadrant_neg",
                {"target_x": _f(-0.06, -0.03), "target_y": _f(-0.06, -0.03)},
                {"target_x": "-0.0347", "target_y": "-0.0429"},
                "302d09670ddb8785079e65e76d31da8680de1c0dc882aa34d7c5bdb53a1626f7",
            ),
        ],
    )


def _handover_pinned_box(bias_deg: float, sigma_deg: float) -> dict[str, ParamRange]:
    """A prereg-pinned presenter box (degenerate edges = the tagged Gate-0 values).

    Values verbatim from ``spikes/preregistration/handover_place/gate0.yaml``
    (tag ``prereg-handover-place-gate0-rev2-2026-06-26``): matched
    ``grasp_pose_sigma_deg`` 2.0, mismatch sigma 10.0 with the bias
    sweep {15, 30, 45}; lateral is NOT inflated (success-side channel);
    timing skew per the prereg partner block.
    """
    matched = bias_deg == 0.0
    return {
        "lateral_offset_x_m": _f(0.0, 0.0),
        "lateral_offset_y_m": _f(0.0, 0.0),
        "lateral_sigma_m": _f(2.0e-4, 2.0e-4),
        "grasp_pose_bias_deg": _f(bias_deg, bias_deg),
        "grasp_pose_sigma_deg": _f(sigma_deg, sigma_deg),
        "timing_skew_bias_s": _f(0.0, 0.0) if matched else _f(0.2, 0.2),
        "timing_skew_sigma_s": _f(0.05, 0.05) if matched else _f(0.1, 0.1),
    }


def _handover_offgrid_box() -> dict[str, ParamRange]:
    """The off-grid presenter box — the preregistered sweep envelope, not a grid point."""
    return {
        "lateral_offset_x_m": _f(0.0, 0.0),
        "lateral_offset_y_m": _f(0.0, 0.0),
        "lateral_sigma_m": _f(1.0e-4, 3.0e-4),
        "grasp_pose_bias_deg": _f(15.0, 45.0),
        "grasp_pose_sigma_deg": _f(8.0, 12.0),
        "timing_skew_bias_s": _f(0.1, 0.3),
        "timing_skew_sigma_s": _f(0.05, 0.15),
    }


@register_partner_set
def handover_place_partners_v1() -> PartnerSetSpec:
    """Handover-place set v1 — presenter variants on the Gate-0 mismatch channel (ADR-009).

    ``handover_place@1`` is ADMITTED (CB-04). Members reuse the frozen
    :class:`chamber.partners.handover_presenter.HandoverPresenterPartner`
    machinery; the four pinned members carry the tagged Gate-0 prereg
    grasp-pose sigma/bias values verbatim (the only mismatch channel —
    lateral is success-side, per the prereg's B1 correction), plus two
    off-grid members drawn from the preregistered sweep envelope.
    Degenerate-box members are value-transparent by construction (the
    prereg is public); the withholding discipline bites only on
    non-degenerate boxes — recorded honestly on the cards.
    """

    def _member(
        name: str,
        box: dict[str, ParamRange],
        split: str,
        params: dict[str, str] | None,
        digest: str,
    ) -> PartnerMemberSpec:
        return PartnerMemberSpec(
            member_name=name,
            registry_class="handover_presenter",
            role="presenter",
            split=split,  # type: ignore[arg-type]  # validated Literal by pydantic
            param_box=box,
            params=params,
            params_sha256=digest,
        )

    return PartnerSetSpec(
        set_id="handover_place_partners",
        version=1,
        task_id="handover_place",
        task_version=1,
        floor=0.90,
        floor_probe="free_regrasp",
        probe_seeds=list(PROBE_SEEDS_V1),
        probe_episodes_per_seed=PROBE_EPISODES_PER_SEED_V1,
        notes=(
            "Scripted stratum of the handover_place@1 set (ADR-009 as "
            "amended). Fingerprint probe at the canonical coupling-valid "
            "anchor cell (clearance 0.2, fast basis, takt 1.5 s — the CB-04 "
            "A2 committed env numbers); competence floor at the Gate-0 "
            "free-re-grasp endpoint, where budget pressure is off and only "
            "a wall scores low. The anchor-cell success spread across "
            "members (matched ~1.0 down to bias-45 ~0.25) is the measured "
            "coupling, not incompetence."
        ),
        members=[
            _member(
                "presenter_matched",
                _handover_pinned_box(0.0, 2.0),
                "private",
                None,
                "d4f8fcf58e750c8dcd223ff4e78cb08da29f758d8b2fb03b9886789711915123",
            ),
            _member(
                "presenter_mismatch_15",
                _handover_pinned_box(15.0, 10.0),
                "public",
                {
                    "grasp_pose_bias_deg": "15.0",
                    "grasp_pose_sigma_deg": "10.0",
                    "lateral_offset_x_m": "0.0",
                    "lateral_offset_y_m": "0.0",
                    "lateral_sigma_m": "0.0002",
                    "timing_skew_bias_s": "0.2",
                    "timing_skew_sigma_s": "0.1",
                },
                "21dad7ccf5614c6990953b65b0287d41deeb18709d40647555c210a65f75850e",
            ),
            _member(
                "presenter_mismatch_30",
                _handover_pinned_box(30.0, 10.0),
                "public",
                {
                    "grasp_pose_bias_deg": "30.0",
                    "grasp_pose_sigma_deg": "10.0",
                    "lateral_offset_x_m": "0.0",
                    "lateral_offset_y_m": "0.0",
                    "lateral_sigma_m": "0.0002",
                    "timing_skew_bias_s": "0.2",
                    "timing_skew_sigma_s": "0.1",
                },
                "8a14dce84204dd4fad781eaa34aace8d037439a0b74ecc57e2d16ac72c3397a2",
            ),
            _member(
                "presenter_mismatch_45",
                _handover_pinned_box(45.0, 10.0),
                "public",
                {
                    "grasp_pose_bias_deg": "45.0",
                    "grasp_pose_sigma_deg": "10.0",
                    "lateral_offset_x_m": "0.0",
                    "lateral_offset_y_m": "0.0",
                    "lateral_sigma_m": "0.0002",
                    "timing_skew_bias_s": "0.2",
                    "timing_skew_sigma_s": "0.1",
                },
                "4ad220f5eb49a4c94449ac90c074b7694c2f18f86ff72a01ad38c16d08c412f5",
            ),
            _member(
                "presenter_offgrid_a",
                _handover_offgrid_box(),
                "public",
                {
                    "grasp_pose_bias_deg": "15.6982",
                    "grasp_pose_sigma_deg": "11.661",
                    "lateral_offset_x_m": "0.0",
                    "lateral_offset_y_m": "0.0",
                    "lateral_sigma_m": "0.0002",
                    "timing_skew_bias_s": "0.2855",
                    "timing_skew_sigma_s": "0.1242",
                },
                "d4b9324e9d4f185ee33f3539bd37ccf83bf0ae8a516e16a9432437c4f19d16a0",
            ),
            _member(
                "presenter_offgrid_b",
                _handover_offgrid_box(),
                "public",
                {
                    "grasp_pose_bias_deg": "18.3353",
                    "grasp_pose_sigma_deg": "11.9689",
                    "lateral_offset_x_m": "0.0",
                    "lateral_offset_y_m": "0.0",
                    "lateral_sigma_m": "0.0003",
                    "timing_skew_bias_s": "0.1793",
                    "timing_skew_sigma_s": "0.1134",
                },
                "493a16d55d28b33593482461eecb2dbeb77d5ff505ba4d3c6cb28dbdecfbb77d",
            ),
        ],
    )


__all__ = [
    "PROBE_EPISODES_PER_SEED_V1",
    "PROBE_SEEDS_V1",
    "cocarry_partners_v1",
    "cocarry_partners_v2",
    "handover_place_partners_v1",
    "stage1_pickplace_as_partners_v1",
]
