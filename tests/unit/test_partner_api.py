# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``chamber.partners.api`` (T4.1).

Covers ADR-009 §Decision (Protocol shape, PartnerSpec) and ADR-006 risk #3 /
ADR-004 §risk-mitigation #2 (partner_id stable hash).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from chamber.partners import api as partners_api
from chamber.partners.api import FrozenPartner, PartnerSpec


class TestPartnerSpec:
    def test_partner_id_is_16_hex_chars(self) -> None:
        """ADR-004 §risk-mitigation #2: 16-char hex (= 64-bit) hash."""
        spec = PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
        )
        assert re.fullmatch(r"[0-9a-f]{16}", spec.partner_id)

    def test_partner_id_stable_across_construction(self) -> None:
        """ADR-006 risk #3: identical specs hash to the same partner_id."""
        a = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="local://a")
        b = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="local://a")
        assert a.partner_id == b.partner_id

    def test_partner_id_changes_when_class_name_changes(self) -> None:
        """Identity differentiator: class_name."""
        a = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="u")
        b = PartnerSpec(class_name="y", seed=1, checkpoint_step=10, weights_uri="u")
        assert a.partner_id != b.partner_id

    def test_partner_id_changes_when_seed_changes(self) -> None:
        """Identity differentiator: seed."""
        a = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="u")
        b = PartnerSpec(class_name="x", seed=2, checkpoint_step=10, weights_uri="u")
        assert a.partner_id != b.partner_id

    def test_partner_id_changes_when_checkpoint_step_changes(self) -> None:
        """Identity differentiator: checkpoint_step (50%-reward vs converged)."""
        a = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="u")
        b = PartnerSpec(class_name="x", seed=1, checkpoint_step=20, weights_uri="u")
        assert a.partner_id != b.partner_id

    def test_partner_id_changes_when_weights_uri_changes(self) -> None:
        """Identity differentiator: weights_uri (different LoRA fine-tunes)."""
        a = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="u1")
        b = PartnerSpec(class_name="x", seed=1, checkpoint_step=10, weights_uri="u2")
        assert a.partner_id != b.partner_id

    def test_partner_id_excludes_extra(self) -> None:
        """Plan/04 §3.1: ``extra`` carries task / embodiment metadata only.

        Two specs that differ only in ``extra`` describe the same logical
        partner (same weights, same training history) — the conformal filter
        must not re-init lambda for an env-config-only change.
        """
        a = PartnerSpec(
            class_name="x",
            seed=1,
            checkpoint_step=10,
            weights_uri="u",
            extra={"task": "stage0_smoke"},
        )
        b = PartnerSpec(
            class_name="x",
            seed=1,
            checkpoint_step=10,
            weights_uri="u",
            extra={"task": "pickplace"},
        )
        assert a.partner_id == b.partner_id

    def test_partner_id_handles_none_checkpoint_and_weights_uri(self) -> None:
        """Scripted partners: checkpoint_step and weights_uri are both None."""
        spec = PartnerSpec(
            class_name="scripted_heuristic",
            seed=0,
            checkpoint_step=None,
            weights_uri=None,
        )
        assert re.fullmatch(r"[0-9a-f]{16}", spec.partner_id)

    def test_partner_spec_is_frozen(self) -> None:
        """Immutability: spec mutations would corrupt the registry's identity contract."""
        import dataclasses

        spec = PartnerSpec(class_name="x", seed=1, checkpoint_step=None, weights_uri=None)
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.seed = 2  # type: ignore[misc]

    def test_partner_id_distinguishes_none_from_string_none(self) -> None:
        """ADR-006 risk #3: ``None`` weights_uri MUST NOT collide with literal ``"None"``.

        A scripted partner has ``weights_uri=None`` while a misconfigured
        Phase-1 partner could ship ``weights_uri="None"``; if the conformal
        filter saw them as the same partner_id it would fail to re-init
        lambda on the swap. The hash uses ``repr`` of a tuple so the two
        cases produce distinct digests.
        """
        scripted = PartnerSpec(class_name="x", seed=0, checkpoint_step=None, weights_uri=None)
        stringy = PartnerSpec(class_name="x", seed=0, checkpoint_step=None, weights_uri="None")
        assert scripted.partner_id != stringy.partner_id

    def test_partner_id_distinguishes_none_from_string_none_on_checkpoint_step(
        self,
    ) -> None:
        """ADR-006 risk #3: same protection for ``checkpoint_step`` field."""
        scripted = PartnerSpec(class_name="x", seed=0, checkpoint_step=None, weights_uri="u")
        # checkpoint_step is typed int | None so we can't pass "None" directly,
        # but the f-string-based hash would have collapsed e.g. None vs 0 if the
        # hash used `str(...)` without distinguishing None. Verify the structural
        # representation also separates None from a literal int that stringifies
        # plausibly.
        zero = PartnerSpec(class_name="x", seed=0, checkpoint_step=0, weights_uri="u")
        assert scripted.partner_id != zero.partner_id

    def test_extra_defaults_to_empty_dict(self) -> None:
        """Default factory: ``extra`` is an empty dict per spec."""
        spec = PartnerSpec(class_name="x", seed=1, checkpoint_step=None, weights_uri=None)
        assert spec.extra == {}


class TestFrozenPartnerProtocol:
    def test_protocol_is_runtime_checkable(self) -> None:
        """ADR-009 §Decision: Protocol is runtime-checkable so adapters can be sniffed."""
        assert hasattr(FrozenPartner, "__instancecheck__")

    def test_protocol_exposes_reset_and_act_only(self) -> None:
        """ADR-009 §Decision: train/learn/update do NOT appear on the Protocol surface.

        The Protocol intentionally omits the forbidden names so type-checked
        callers cannot reach for them; the runtime ``_FORBIDDEN_ATTRS`` shield
        backs that up for non-type-checked callers.
        """
        protocol_methods = {name for name in dir(FrozenPartner) if not name.startswith("_")}
        assert "reset" in protocol_methods
        assert "act" in protocol_methods
        for forbidden in ("train", "learn", "update", "update_params", "fit"):
            assert forbidden not in protocol_methods

    def test_minimal_implementation_satisfies_runtime_check(self) -> None:
        """A class providing reset + act + spec satisfies the Protocol at runtime."""

        class _Minimal:
            spec: PartnerSpec

            def __init__(self) -> None:
                self.spec = PartnerSpec(
                    class_name="x", seed=0, checkpoint_step=None, weights_uri=None
                )

            def reset(self, *, seed: int | None = None) -> None:
                del seed

            def act(self, obs: dict[str, object], *, deterministic: bool = True) -> np.ndarray:
                del obs, deterministic
                return np.zeros(2, dtype=np.float32)

        assert isinstance(_Minimal(), FrozenPartner)


class TestPublicSurface:
    def test_all_listed_symbols_resolve(self) -> None:
        """Every name in ``api.__all__`` is importable from the module."""
        for name in partners_api.__all__:
            assert hasattr(partners_api, name)

    def test_all_contains_expected_symbols(self) -> None:
        """The api module exports the Protocol + the spec dataclass."""
        assert set(partners_api.__all__) == {"FrozenPartner", "PartnerSpec"}
