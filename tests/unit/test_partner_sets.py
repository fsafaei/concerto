# SPDX-License-Identifier: Apache-2.0
"""Tier-1 tests for the partner-set mechanics (ADR-009 §Decision as amended 2026-07-05).

Covers: parameter digests (canonical + process-stable), box-derived
parameter draws, the member identity fold (``member://`` weights_uri →
distinct ``partner_id`` per parameterization), the deterministic
70/30 split rule (recompute → identical; committed drift loud-fails),
set-version resolution, the withheld-parameters scheme (refusal without
the maintainer seed; wrong-seed digest failure), the committed v1 set
composition (co-carry ≥10 members with ≥7 public), and the bounded-lag
impedance member's delayed-but-convergent action emission.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import chamber.partners  # noqa: F401 - registers the v1 sets
from chamber.partners.api import PartnerSpec
from chamber.partners.sets import (
    PRIVATE_PARAMS_ENV,
    ParamRange,
    PartnerMemberSpec,
    PartnerSetSpec,
    WithheldParametersError,
    canonical_params_json,
    compute_split,
    derive_member_params,
    get_partner_set,
    list_partner_sets,
    member_weights_uri,
    params_sha256,
    parse_set_slug,
    private_seed_available,
    public_member_count,
    resolve_member_params,
    resolve_set_members,
)

_TEST_ROOT_SEED = 424242


def _box() -> dict[str, ParamRange]:
    return {
        "kp": ParamRange(lo=1.0, hi=3.0),
        "lag_steps": ParamRange(lo=0, hi=4, kind="int"),
    }


def _member(
    name: str,
    *,
    split: str = "public",
    set_id: str = "test_set",
) -> PartnerMemberSpec:
    params = derive_member_params(set_id, name, _box(), root_seed=_TEST_ROOT_SEED)
    return PartnerMemberSpec(
        member_name=name,
        registry_class="cocarry_impedance",
        role="partner_arm",
        split=split,  # type: ignore[arg-type]
        param_box=_box(),
        params=params if split == "public" else None,
        params_sha256=params_sha256(params),
    )


class TestParamsDigest:
    def test_canonical_json_sorts_keys(self) -> None:
        a = canonical_params_json({"b": "2", "a": "1"})
        b = canonical_params_json({"a": "1", "b": "2"})
        assert a == b == '{"a": "1", "b": "2"}'

    def test_digest_stable_across_processes(self) -> None:
        """SHA-256 over the canonical encoding — no Python hash randomisation."""
        params = {"kp": "2.5", "lag_steps": "3"}
        local = params_sha256(params)
        code = (
            "from chamber.partners.sets import params_sha256;"
            "print(params_sha256({'lag_steps': '3', 'kp': '2.5'}))"
        )
        remote = subprocess.run(  # noqa: S603 - fixed interpreter, fixed code
            [sys.executable, "-c", code],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
        assert remote == local


class TestParamRange:
    def test_inverted_box_rejected(self) -> None:
        with pytest.raises(ValueError, match="lo <= hi"):
            ParamRange(lo=2.0, hi=1.0)

    def test_int_kind_requires_integer_bounds(self) -> None:
        with pytest.raises(ValueError, match="integer bounds"):
            ParamRange(lo=0.5, hi=2.0, kind="int")

    def test_contains(self) -> None:
        edge = ParamRange(lo=1.0, hi=2.0)
        assert edge.contains("1.5")
        assert not edge.contains("2.5")
        assert not edge.contains("nope")
        int_edge = ParamRange(lo=0, hi=3, kind="int")
        assert int_edge.contains("2")
        assert not int_edge.contains("1.5")


class TestDeriveMemberParams:
    def test_deterministic_and_inside_box(self) -> None:
        a = derive_member_params("s", "m", _box(), root_seed=7)
        b = derive_member_params("s", "m", _box(), root_seed=7)
        assert a == b
        assert _box()["kp"].contains(a["kp"])
        assert _box()["lag_steps"].contains(a["lag_steps"])

    def test_distinct_members_draw_independently(self) -> None:
        a = derive_member_params("s", "m1", _box(), root_seed=7)
        b = derive_member_params("s", "m2", _box(), root_seed=7)
        assert a != b

    def test_degenerate_box_pins_the_value(self) -> None:
        box = {"grasp_pose_bias_deg": ParamRange(lo=30.0, hi=30.0)}
        assert derive_member_params("s", "m", box, root_seed=7) == {"grasp_pose_bias_deg": "30.0"}


class TestMemberIdentity:
    def test_distinct_params_distinct_partner_id_same_class(self) -> None:
        """The member:// fold: parameterizations of one class never collide (ADR-006 risk #3)."""
        m1 = _member("m1")
        m2 = _member("m2")
        assert m1.registry_class == m2.registry_class
        assert m1.partner_id != m2.partner_id
        # And distinct from the bare-class ad-hoc spec.
        adhoc = PartnerSpec("cocarry_impedance", 0, None, None)
        assert m1.partner_id != adhoc.partner_id

    def test_partner_id_computable_without_private_params(self) -> None:
        private = _member("m1", split="private")
        assert private.params is None
        assert private.partner_id == _member("m1", split="public").partner_id

    def test_weights_uri_carries_the_digest(self) -> None:
        m = _member("m1")
        assert m.weights_uri == member_weights_uri("cocarry_impedance", m.params_sha256)

    def test_partner_spec_rejects_wrong_params(self) -> None:
        m = _member("m1")
        with pytest.raises(WithheldParametersError, match="refusing to build"):
            m.partner_spec(params={"kp": "9.9", "lag_steps": "0"})


class TestMemberValidators:
    def test_public_member_requires_params(self) -> None:
        good = _member("m1")
        with pytest.raises(ValueError, match="no parameter values"):
            PartnerMemberSpec(
                member_name="m1",
                registry_class="cocarry_impedance",
                role="partner_arm",
                split="public",
                param_box=_box(),
                params=None,
                params_sha256=good.params_sha256,
            )

    def test_private_member_must_withhold_params(self) -> None:
        good = _member("m1")
        assert good.params is not None
        with pytest.raises(ValueError, match="withheld"):
            PartnerMemberSpec(
                member_name="m1",
                registry_class="cocarry_impedance",
                role="partner_arm",
                split="private",
                param_box=_box(),
                params=good.params,
                params_sha256=good.params_sha256,
            )

    def test_params_outside_box_rejected(self) -> None:
        good = _member("m1")
        assert good.params is not None
        bad = {**good.params, "kp": "99.0"}
        with pytest.raises(ValueError, match="outside"):
            PartnerMemberSpec(
                member_name="m1",
                registry_class="cocarry_impedance",
                role="partner_arm",
                split="public",
                param_box=_box(),
                params=bad,
                params_sha256=params_sha256(bad),
            )

    def test_digest_mismatch_rejected(self) -> None:
        good = _member("m1")
        with pytest.raises(ValueError, match="params_sha256"):
            PartnerMemberSpec(
                member_name="m1",
                registry_class="cocarry_impedance",
                role="partner_arm",
                split="public",
                param_box=_box(),
                params=good.params,
                params_sha256="0" * 64,
            )


class TestSplitRule:
    def test_public_member_count_is_ceil(self) -> None:
        assert public_member_count(3) == 3
        assert public_member_count(6) == 5
        assert public_member_count(10) == 7
        assert public_member_count(11) == 8

    def test_recompute_is_order_invariant(self) -> None:
        ids = [f"{i:016x}" for i in (9, 1, 5, 3, 7, 2, 8, 4, 6, 0)]
        forward = compute_split(ids)
        shuffled = compute_split(list(reversed(ids)))
        assert forward == shuffled
        assert sum(1 for v in forward.values() if v == "public") == 7
        # Ascending hash order: the lowest ⌈0.7·N⌉ hashes are public.
        assert forward[f"{0:016x}"] == "public"
        assert forward[f"{9:016x}"] == "private"

    def test_duplicate_ids_rejected(self) -> None:
        with pytest.raises(ValueError, match="distinct"):
            compute_split(["a" * 16, "a" * 16])

    def test_set_spec_rejects_hand_picked_split(self) -> None:
        """A committed split that disagrees with the rule is a registration error."""
        members = [_member(f"m{i}") for i in range(3)]
        # N=3 → all public under ⌈0.7·N⌉; flipping one to private must fail.
        flipped = PartnerMemberSpec(
            member_name=members[0].member_name,
            registry_class=members[0].registry_class,
            role=members[0].role,
            split="private",
            param_box=members[0].param_box,
            params=None,
            params_sha256=members[0].params_sha256,
        )
        with pytest.raises(ValueError, match="no hand-picking"):
            PartnerSetSpec(
                set_id="test_set",
                version=1,
                task_id="cocarry",
                task_version=1,
                floor=0.5,
                probe_seeds=[0],
                probe_episodes_per_seed=1,
                members=[flipped, members[1], members[2]],
            )


class TestSetRegistry:
    def test_registered_sets(self) -> None:
        assert list_partner_sets() == [
            "cocarry_partners@v1",
            "cocarry_partners@v2",
            "handover_place_partners@v1",
            "stage1_pickplace_as_partners@v1",
        ]

    def test_version_resolution(self) -> None:
        # Unversioned resolution returns the LATEST version; pinning
        # still reaches v1 (ADR-027 §Versioning: old bundles reference
        # the version they ran against).
        latest = get_partner_set("cocarry_partners")
        assert latest.slug == "cocarry_partners@v2"
        pinned = get_partner_set("cocarry_partners", version=1)
        assert pinned.slug == "cocarry_partners@v1"
        assert latest is not pinned

    def test_unknown_id_and_version_loud_fail(self) -> None:
        with pytest.raises(KeyError, match="registered set ids"):
            get_partner_set("nope")
        with pytest.raises(KeyError, match="registered versions"):
            get_partner_set("cocarry_partners", version=9)

    def test_parse_set_slug(self) -> None:
        assert parse_set_slug("cocarry_partners@v2") == ("cocarry_partners", 2)
        assert parse_set_slug("cocarry_partners") == ("cocarry_partners", None)


class TestWithheldParameters:
    @pytest.fixture
    def synthetic_set(self) -> PartnerSetSpec:
        members = [
            _member(f"m{i}", set_id="withheld_set") for i in range(4)
        ]  # N=4 → 3 public, 1 private by the rule
        split = compute_split([m.partner_id for m in members])
        rebuilt = [
            PartnerMemberSpec(
                member_name=m.member_name,
                registry_class=m.registry_class,
                role=m.role,
                split=split[m.partner_id],
                param_box=m.param_box,
                params=m.params if split[m.partner_id] == "public" else None,
                params_sha256=m.params_sha256,
            )
            for m in members
        ]
        return PartnerSetSpec(
            set_id="withheld_set",
            version=1,
            task_id="cocarry",
            task_version=1,
            floor=0.5,
            probe_seeds=[0],
            probe_episodes_per_seed=1,
            members=rebuilt,
        )

    def test_public_resolution_needs_no_seed(
        self, synthetic_set: PartnerSetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(PRIVATE_PARAMS_ENV, raising=False)
        assert not private_seed_available()
        resolved = resolve_set_members(synthetic_set)
        assert len(resolved) == 3

    def test_include_private_refused_without_seed(
        self, synthetic_set: PartnerSetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(PRIVATE_PARAMS_ENV, raising=False)
        with pytest.raises(WithheldParametersError, match="withheld"):
            resolve_set_members(synthetic_set, include_private=True)

    def test_include_private_with_seed_verifies_digest(
        self, synthetic_set: PartnerSetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PRIVATE_PARAMS_ENV, str(_TEST_ROOT_SEED))
        resolved = resolve_set_members(synthetic_set, include_private=True)
        assert len(resolved) == 4
        (private_member,) = [m for m in synthetic_set.members if m.split == "private"]
        params = resolve_member_params(synthetic_set, private_member)
        assert params_sha256(params) == private_member.params_sha256

    def test_wrong_seed_loud_fails(
        self, synthetic_set: PartnerSetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PRIVATE_PARAMS_ENV, "1")
        with pytest.raises(WithheldParametersError, match="wrong maintainer seed"):
            resolve_set_members(synthetic_set, include_private=True)

    def test_malformed_seed_loud_fails(
        self, synthetic_set: PartnerSetSpec, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(PRIVATE_PARAMS_ENV, "not-an-int")
        assert not private_seed_available()
        with pytest.raises(WithheldParametersError, match="integer root seed"):
            resolve_set_members(synthetic_set, include_private=True)


class TestV1Composition:
    """The committed v1 rosters (ADR-009 as amended acceptance: ≥10 members, ≥7 public)."""

    def test_cocarry_set_size_and_split(self) -> None:
        s = get_partner_set("cocarry_partners", version=1)
        assert len(s.members) >= 10
        assert len(s.public_members) >= 7
        assert s.task_id == "cocarry"
        assert s.floor == 0.75

    def test_control_and_handover_sets(self) -> None:
        control = get_partner_set("stage1_pickplace_as_partners", version=1)
        assert len(control.members) == 3
        assert not control.private_members  # ⌈0.7·3⌉ = 3 → all public
        handover = get_partner_set("handover_place_partners", version=1)
        assert len(handover.members) == 6
        assert len(handover.public_members) == 5
        assert handover.floor_probe == "free_regrasp"

    def test_split_recomputes_identically(self) -> None:
        for slug in list_partner_sets():
            set_id, version = parse_set_slug(slug)
            s = get_partner_set(set_id, version=version)
            expected = compute_split([m.partner_id for m in s.members])
            assert {m.member_name: m.split for m in s.members} == {
                m.member_name: expected[m.partner_id] for m in s.members
            }

    def test_partner_ids_unique_within_and_across_sets(self) -> None:
        ids: list[tuple[str, str]] = []
        seen_by_set: dict[str, str] = {}
        for slug in list_partner_sets():
            set_id, version = parse_set_slug(slug)
            for m in get_partner_set(set_id, version=version).members:
                # Unique within a set version, and across DIFFERENT
                # set_ids; the same identity recurring across versions
                # of ONE set is the ADR-027 §Versioning contract
                # (member identity is version-stable by design).
                key = f"{set_id}:{m.partner_id}"
                assert key not in seen_by_set or seen_by_set[key] == m.member_name, (
                    f"{slug}: partner_id {m.partner_id} collides across members"
                )
                seen_by_set[key] = m.member_name
                ids.append((slug, m.partner_id))
        for slug in list_partner_sets():
            within = [pid for s, pid in ids if s == slug]
            assert len(set(within)) == len(within)

    def test_registered_classes_exist(self) -> None:
        from chamber.partners.registry import list_registered

        registered = set(list_registered())
        for slug in list_partner_sets():
            set_id, version = parse_set_slug(slug)
            for m in get_partner_set(set_id, version=version).members:
                assert m.registry_class in registered

    def test_quarantined_wall_stays_out(self) -> None:
        """The co-insert Gate-0 lesson: static_override is a wall, not a partner (CB-01)."""
        for slug in list_partner_sets():
            set_id, version = parse_set_slug(slug)
            for m in get_partner_set(set_id, version=version).members:
                assert "static_override" not in m.registry_class


class TestBoundedLagMember:
    """The lag_steps extension (ADR-009 as amended: sluggish but competent)."""

    @staticmethod
    def _partner(lag: str):
        from chamber.partners.registry import load_partner

        extra = {
            "uid": "panda_wristcam",
            "base_xyz": "-0.5,0,0",
            "base_yaw_deg": "0",
            "end_sign": "1",
            "bar_half_len": "0.115",
            "lag_steps": lag,
        }
        return load_partner(PartnerSpec("cocarry_impedance", 0, None, None, extra))

    @staticmethod
    def _obs() -> dict:
        q = np.array([0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, np.pi * 3 / 4, -np.pi / 4])
        return {
            "agent": {
                "panda_wristcam": {"qpos": np.concatenate([q, [0.04, 0.04]]).astype(np.float32)}
            },
            "extra": {"goal_pos": np.array([0.0, 0.1, 0.3], dtype=np.float32)},
        }

    def test_lag_zero_is_byte_identical_path(self) -> None:
        baseline = self._partner("0")
        implicit = self._partner("0")
        del implicit  # constructing with the default exercises the same path
        baseline.reset(seed=0)
        a = baseline.act(self._obs())
        assert float(np.linalg.norm(a[:7])) > 1e-3

    def test_lagged_member_emits_delayed_commands(self) -> None:
        lag = 3
        lagged = self._partner(str(lag))
        plain = self._partner("0")
        lagged.reset(seed=0)
        plain.reset(seed=0)
        obs = self._obs()
        plain_first = plain.act(obs)
        # Warm-up: the lagged member holds still (gripper-only action).
        for _ in range(lag):
            held = lagged.act(obs)
            np.testing.assert_array_equal(held[:7], np.zeros(7, dtype=np.float32))
            assert held[7] == pytest.approx(1.0)
        # Step lag+1 emits the command computed at step 1 (same obs → equal).
        delayed = lagged.act(obs)
        np.testing.assert_allclose(delayed[:7], plain_first[:7], atol=1e-7)

    def test_reset_clears_the_lag_buffer(self) -> None:
        lagged = self._partner("2")
        lagged.reset(seed=0)
        obs = self._obs()
        lagged.act(obs)
        lagged.act(obs)
        lagged.reset(seed=1)
        held = lagged.act(obs)
        np.testing.assert_array_equal(held[:7], np.zeros(7, dtype=np.float32))

    def test_lag_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="lag_steps"):
            self._partner("11")
        with pytest.raises(ValueError, match="lag_steps"):
            self._partner("-1")
        with pytest.raises(ValueError, match="lag_steps"):
            self._partner("two")


class TestLearnedStratumV2:
    """The v2 learned-member machinery (ADR-027 §Versioning; ADR-011 §Decision as amended)."""

    _SHA = "a" * 64
    _EMPTY_DIGEST = params_sha256({})

    def _learned(self, **overrides: object) -> PartnerMemberSpec:
        base: dict[str, object] = {
            "member_name": "joint_s9",
            "registry_class": "frozen_cocarry_joint",
            "role": "partner_arm",
            "split": "public",
            "seed": 9,
            "checkpoint_step": 150_000,
            "param_box": {},
            "params": {},
            "params_sha256": self._EMPTY_DIGEST,
            "checkpoint_uri": "local://artifacts/pair.pt",
            "checkpoint_sha256": self._SHA,
            "provenance": (
                "trained jointly with the ego actor of pair checkpoint sha256 " + self._SHA
            ),
        }
        base.update(overrides)
        return PartnerMemberSpec.model_validate(base)

    def test_weights_uri_is_the_real_checkpoint(self) -> None:
        member = self._learned()
        assert member.weights_uri == "local://artifacts/pair.pt"
        spec = member.partner_spec(params={}, seat_extra={"uid": "a", "other_uid": "b"})
        assert spec.weights_uri == "local://artifacts/pair.pt"
        assert spec.checkpoint_step == 150_000
        assert spec.seed == 9

    def test_checkpoint_custody_invariants(self) -> None:
        with pytest.raises(ValueError, match="checkpoint_sha256"):
            self._learned(checkpoint_sha256=None)
        with pytest.raises(ValueError, match="checkpoint_step"):
            self._learned(checkpoint_step=None)
        with pytest.raises(ValueError, match="unanchored"):
            self._learned(checkpoint_uri=None)

    def test_scripted_members_unchanged(self) -> None:
        v1 = get_partner_set("cocarry_partners", version=1)
        v2 = get_partner_set("cocarry_partners", version=2)
        v1_ids = {m.member_name: m.partner_id for m in v1.members}
        v2_ids = {m.member_name: m.partner_id for m in v2.members}
        # Identity stability across the version bump (ADR-027 §Versioning).
        for name, pid in v1_ids.items():
            assert v2_ids[name] == pid
        # Scripted members still ride the member:// URI.
        nominal = next(m for m in v2.members if m.member_name == "imp_nominal")
        assert nominal.weights_uri.startswith("member://")

    def test_v2_roster_and_split(self) -> None:
        v2 = get_partner_set("cocarry_partners", version=2)
        # 11 scripted + the ONE floor-passing learned member (four of
        # the five preregistered jointly-trained candidates failed the
        # 0.75 cross-play floor and were dropped per the set rule).
        assert len(v2.members) == 12
        assert len(v2.public_members) == 9
        assert len(v2.private_members) == 3
        learned = [m for m in v2.members if m.checkpoint_uri is not None]
        assert [m.member_name for m in learned] == ["joint_s4"]
        assert "trained jointly with" in learned[0].provenance
        assert learned[0].split == "public"
        # Every v1 split label is reproduced over the 12-member roster.
        v1 = get_partner_set("cocarry_partners", version=1)
        v1_split = {m.member_name: m.split for m in v1.members}
        for m in v2.members:
            if m.member_name in v1_split:
                assert m.split == v1_split[m.member_name]
