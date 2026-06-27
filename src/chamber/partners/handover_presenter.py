# SPDX-License-Identifier: Apache-2.0
"""Frozen scripted presenter partner for the Gate-0 handover-and-place spike.

Phase-2, non-gating ADR-026 research spike (invariant I1). The presenter hands a part
into the shared workspace and then **lets go**; it binds the ego ONLY through the
initial condition it hands over. Per the executor-prompt Rev 2 mechanism correction
(B1), that condition has two sub-channels which behave differently, and the mismatch
lives in exactly one of them:

* **Lateral position offset** — a competent six-axis ego corrects this by translating
  its place motion (cheap, no re-grasp), so it is the downstream *success tolerance*,
  NOT the coupling channel. Both variants present near-identical (small) lateral
  offsets.
* **Grasp-pose / orientation error** — how the part is oriented in the ego's gripper.
  Beyond the wrist-correction range the ego must set the part down and re-grasp; beyond
  the re-acquire range even a re-grasp leaves a residual. THIS is the coupling channel,
  and the only thing the two variants differ in.

So ``matched`` and ``mismatched`` differ ONLY in the grasp-pose distribution. There is
deliberately NO contact-impedance / force-coupling term (the partner has let go; that
is the co-hold mechanism and would exercise the wrong channel -> uninformative null).

Black-box contract (ADR-009 §Decision / §Consequences). Frozen
:class:`chamber.partners.interface.PartnerBase` subclass: ``reset`` and ``act`` only;
``PartnerBase.__getattr__`` blocks every ``_FORBIDDEN_ATTRS`` name, so no joint
training and no policy-weight access is reachable. The ego consumes the presented part
pose + grasp-pose (allowed: pose visibility is permitted under the ADR-009 2026-05-21
amendment; policy access is not); it never reads this partner's policy.

Determinism (P6 / ADR-002). The per-episode presentation is a deterministic function
of the reset seed: :meth:`reset` builds a fresh
:func:`concerto.training.seeding.derive_substream` generator and :meth:`act` draws once
from it; across-episode variation is carried by distinct reset seeds, never ad-hoc
``np.random``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.envs.handover_place import HANDOVER_PRESENTATION_DIM
from chamber.partners.interface import PartnerBase
from chamber.partners.registry import register_partner
from concerto.training.seeding import derive_substream

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from chamber.partners.api import PartnerSpec

#: Registry key for the scripted presenter.
HANDOVER_PRESENTER_CLASS: str = "handover_presenter"

#: Keys read from ``spec.extra`` (all stringified floats; ``variant`` is a label). The
#: runner fills these from the committed Gate-0 prereg; the helpers below carry only
#: NON-BINDING placeholder defaults so the partner is unit-testable.
_PARAM_KEYS: tuple[str, ...] = (
    "lateral_offset_x_m",
    "lateral_offset_y_m",
    "lateral_sigma_m",
    "grasp_pose_bias_deg",
    "grasp_pose_sigma_deg",
    "timing_skew_bias_s",
    "timing_skew_sigma_s",
)

#: NON-BINDING placeholder distributions (unit tests only). matched and mismatched
#: differ ONLY in the grasp-pose channel; lateral + timing are the same. The binding
#: values are the pre-registered, externally-anchored numbers injected by the runner.
_PLACEHOLDER_MATCHED: dict[str, float] = {
    "lateral_offset_x_m": 0.0,
    "lateral_offset_y_m": 0.0,
    "lateral_sigma_m": 2.0e-4,
    "grasp_pose_bias_deg": 0.0,
    "grasp_pose_sigma_deg": 1.0,
    "timing_skew_bias_s": 0.0,
    "timing_skew_sigma_s": 0.02,
}
_PLACEHOLDER_MISMATCHED: dict[str, float] = {
    "lateral_offset_x_m": 0.0,  # lateral is NOT the mismatch channel
    "lateral_offset_y_m": 0.0,
    "lateral_sigma_m": 2.0e-4,
    "grasp_pose_bias_deg": 25.0,  # the mismatch: grasp-pose error past the wrist range
    # (placeholder; the binding bias is a Stage-0-derived multiple of the matched sigma)
    "grasp_pose_sigma_deg": 8.0,
    "timing_skew_bias_s": 0.0,
    "timing_skew_sigma_s": 0.05,
}


def presenter_spec(variant: str, *, seed: int = 0, **params: float) -> PartnerSpec:
    """Build a frozen presenter :class:`PartnerSpec` for one variant (ADR-009; ADR-026).

    ``variant`` is ``"matched"`` or ``"mismatched"`` (a provenance label). Any
    presentation parameter in ``_PARAM_KEYS`` may be overridden via ``params``;
    unspecified keys fall back to the NON-BINDING placeholder distribution for that
    variant so the partner is constructible in unit tests. The spike runner overrides
    every key with the committed Gate-0 prereg values.
    """
    from chamber.partners.api import PartnerSpec

    if variant not in ("matched", "mismatched"):
        raise ValueError(f"variant must be 'matched' or 'mismatched', got {variant!r}")
    base = _PLACEHOLDER_MATCHED if variant == "matched" else _PLACEHOLDER_MISMATCHED
    merged = {**base, **params}
    extra = {"variant": variant}
    extra.update({key: repr(float(merged[key])) for key in _PARAM_KEYS})
    return PartnerSpec(
        class_name=HANDOVER_PRESENTER_CLASS,
        seed=seed,
        checkpoint_step=None,
        weights_uri=None,
        extra=extra,
    )


@register_partner(HANDOVER_PRESENTER_CLASS)
class HandoverPresenterPartner(PartnerBase):
    """Scripted black-box presenter for handover-and-place (ADR-009; ADR-026 §Decision).

    Emits a presentation action ``[lat_offset_x, lat_offset_y, grasp_pose_error_deg,
    timing_skew_s]`` drawn from its distribution; the matched/mismatched split lives in
    the grasp-pose term only. Frozen: implements ``reset``/``act`` only;
    ``PartnerBase`` blocks every ``_FORBIDDEN_ATTRS`` lookup so the AHT
    no-joint-training constraint holds at runtime (ADR-009 §Consequences).
    """

    def __init__(self, spec: PartnerSpec) -> None:
        """Bind the variant label and presentation params from ``spec`` (ADR-009)."""
        super().__init__(spec)
        self.variant: str = spec.extra.get("variant", "matched")
        self._params: dict[str, float] = {
            key: float(spec.extra[key]) for key in _PARAM_KEYS if key in spec.extra
        }
        self._rng: np.random.Generator | None = None

    def reset(self, *, seed: int | None = None) -> None:
        """Seed the per-episode presentation RNG (P6 / ADR-002; FrozenPartner contract).

        Builds a fresh deterministic generator so the presentation is a pure function
        of ``seed``; clears any prior episode state.
        """
        episode_seed = 0 if seed is None else int(seed)
        self._rng = derive_substream(
            f"partner.handover_presenter.{self.variant}", root_seed=episode_seed
        ).default_rng()

    def act(self, obs: Mapping[str, Any], *, deterministic: bool = True) -> NDArray[np.floating]:
        """Draw the presentation handed to the ego (ADR-026 §Decision; ADR-009).

        A one-shot draw from the variant's distribution; ``obs`` is unused (the
        presenter conditions on nothing the ego controls). ``deterministic`` is
        accepted for the FrozenPartner Protocol; the draw is fully seed-determined, so
        the flag does not change behaviour.
        """
        del obs, deterministic
        if self._rng is None:
            raise RuntimeError("HandoverPresenterPartner.act called before reset()")
        rng = self._rng
        p = self._params
        lateral_sigma = p.get("lateral_sigma_m", 0.0)
        lat_x = p.get("lateral_offset_x_m", 0.0) + float(rng.normal(0.0, lateral_sigma))
        lat_y = p.get("lateral_offset_y_m", 0.0) + float(rng.normal(0.0, lateral_sigma))
        grasp_pose_error = p.get("grasp_pose_bias_deg", 0.0) + float(
            rng.normal(0.0, p.get("grasp_pose_sigma_deg", 0.0))
        )
        timing_skew = max(
            0.0,
            p.get("timing_skew_bias_s", 0.0)
            + float(rng.normal(0.0, p.get("timing_skew_sigma_s", 0.0))),
        )
        action = np.zeros(HANDOVER_PRESENTATION_DIM, dtype=np.float64)
        action[:] = (lat_x, lat_y, grasp_pose_error, timing_skew)
        return action
