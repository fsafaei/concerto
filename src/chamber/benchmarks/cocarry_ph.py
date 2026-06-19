# SPDX-License-Identifier: Apache-2.0
r"""Co-carry Rung-3 policy-heterogeneity (PH) measurement harness (ADR-026 §Decision 4; ADR-009).

Rung 3 (R-2026-06-B §15) is **the heterogeneity measurement the ladder
exists to produce**: hold the held-out-validated Rung-2 incumbent **frozen**
(:mod:`chamber.benchmarks.cocarry_incumbent`,
:mod:`chamber.benchmarks.cocarry_freeze`) and measure whether pairing it with
a **different-but-capability-matched partner policy** (same Panda body;
:mod:`chamber.partners.cocarry_policy_shift`) degrades cooperation, relative
to the matched reference (frozen incumbent + the matched
``cocarry_impedance`` partner). The drop Δ — or its absence — is the result.

This module is the harness that produces it. It deliberately keeps three
concerns separate so the load-bearing logic is Tier-1-testable without SAPIEN:

1. **Pure statistics** (Tier-1; no SAPIEN). The capability-gate decision, the
   paired Δ, the cluster-bootstrap one-sided CI, the IQM, the Δ pooling, the
   pre-committed decision rule, and the cluster-robust binomial confirmatory
   are pure functions over plain ``{seed: bool}`` maps — so the Tier-1 tests
   pin the inference the pre-registration locks, on synthetic outcomes.
2. **Rollout** (Tier-2; SAPIEN-gated). A single generic two-controller
   rollout (:func:`rollout_pair`) drives the ego seat with any ``obs ->
   action`` closure and the partner seat with any
   :class:`~chamber.partners.api.FrozenPartner`, capturing the per-episode
   **conjunct breakdown** (placed / level / unstressed / static) so a drop's
   mechanism is visible (R-2026-06-B §15 Rung 3 reporting requirement).
3. **Conditions** (Tier-2). The matched reference, the capability calibration
   (candidate + a **cooperative reference** ego — the matched impedance ego,
   *not* the frozen incumbent), and the shifted measurement (frozen incumbent
   + each calibrated teammate), all on the *same* env path
   (:func:`chamber.envs.cocarry_obs.make_cocarry_training_env`, matched
   condition) so only who drives each seat changes.

Governance (binding — R-2026-06-B §15; ADR-026 §Decision 4):

- **Frozen incumbent, never retrained.** The ego seat in the reference and
  shifted conditions is the SHA-verified frozen checkpoint loaded via
  :func:`chamber.benchmarks.cocarry_incumbent.load_frozen_incumbent`.
- **Capability gate defuses the weaker-teammate confound.** A candidate
  enters the test only if, paired with the cooperative reference, it clears
  :data:`C_MIN` (grounded on the matched reference; see
  :data:`C_MIN_DERIVATION`). Exclusions are archived; with exclusions present,
  a Δ≈0 may NOT be read as "axis not coupling-valid" (the gate truncates
  heterogeneity from above and biases Δ toward zero — :data:`NULL_CAVEAT`).
- **Pre-register before measuring.** :data:`C_MIN`, :data:`DELTA_MIN`, the
  seed sets, the decision rule, and the pooling method are module constants
  fixed before the measurement and committed as a JSON pre-registration with a
  git tag (mirrors the Rung-2 freeze prereg, commit 5fc291b). No schema bump
  (I9): a new co-carry tag reusing the existing serialisation surface.
- **No bar-moving.** The success predicate, ``f_max`` (130.6 N), and the
  0.10 m goal radius are unchanged; this module imports them, never redefines
  them.

References:
- ADR-026 §Decision 4 (the Phase-2 forward-design ladder; Rung 3 = PH).
- ADR-009 §Decision (frozen black-box partner; partner-agnostic teammates).
- R-2026-06-B §15 Rung 3 (the teammate set, capability gate, Δ_min, decision
  rule, pooling, and the pre-committed null rule restated as constants here).
- :mod:`chamber.benchmarks.cocarry_incumbent` (the frozen-incumbent load +
  matched eval this extends), :mod:`chamber.benchmarks.cocarry_runner`
  (``EpisodeMetrics`` / ``summarize`` reused), :mod:`chamber.partners.cocarry_policy_shift`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from chamber.benchmarks.cocarry_runner import _to_float
from chamber.envs.cocarry import COCARRY_DEFAULT_EPISODE_LENGTH, cocarry_matched_controller_specs
from chamber.partners.api import PartnerSpec
from chamber.partners.registry import load_partner

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping
    from pathlib import Path

    import gymnasium as gym
    from numpy.typing import NDArray

    from chamber.partners.api import FrozenPartner
    from concerto.training.config import EgoAHTConfig

# Import for the @register_partner side effects so the shifted teammates +
# the matched impedance reference are resolvable when this module drives the
# measurement (scripts / tests).
import chamber.partners.cocarry_impedance
import chamber.partners.cocarry_policy_shift  # noqa: F401

# ---------------------------------------------------------------------------
# Pre-registration-grade constants (R-2026-06-B §15 Rung 3). FIXED before any
# reference / shifted measurement and committed under a git tag. Changing one
# of these after the prereg tag is the forking-path anti-pattern the ladder
# forbids (mirrors the Rung-2 freeze constants in cocarry_freeze.py).
# ---------------------------------------------------------------------------

#: The matched reference partner — the frozen incumbent's cooperation ceiling
#: (frozen incumbent + this partner ≈ 100%, the Rung-2 manifest). UNCHANGED.
MATCHED_PARTNER_CLASS: str = "cocarry_impedance"

#: The cooperative-reference ego used by the capability gate — a hand-written
#: cooperative controller (the matched impedance ego seat), **not** the frozen
#: incumbent (R-2026-06-B §15 Rung 3: the gate must not use the incumbent, or
#: it could not separate "incompetent teammate" from "the incumbent dislikes
#: this teammate"). Pairing a candidate with this proven cooperative ego
#: measures the candidate's *capability*, defusing the weaker-teammate confound.
COOPERATIVE_REFERENCE_EGO_CLASS: str = "cocarry_impedance"

#: Capability-gate threshold: a candidate enters the PH test iff its joint
#: success rate paired with the cooperative reference is >= this.
C_MIN: float = 0.75

#: How :data:`C_MIN` is derived (fixed at pre-statement; archived in the
#: calibration roster). Grounded on the matched reference, NOT chosen to pass
#: teammates.
C_MIN_DERIVATION: str = (
    "Grounded on the matched cooperative reference, which scores ~1.0 joint "
    "success (Rung-1 matched competence; Rung-2 manifest 24/24 on held-out V). "
    "C_min = matched_reference - 0.25 competence margin = 0.75: a candidate "
    "must jointly succeed with the proven cooperative controller on >= 9/12 "
    "calibration seeds. The 0.25 margin admits legitimately-different-but-"
    "competent policies (so the gate does not truncate the heterogeneity it "
    "is meant to admit) while excluding teammates that simply cannot do the "
    "task (success ~0.5 or below, near the single-arm positive-control floor). "
    "Fixed before measurement; the matched teammate's calibrated score M is "
    "reported and C_min = max(0.75, M - 0.25) holds with M ~ 1.0."
)

#: The pre-set band (around the matched teammate's calibrated score M ~ 1.0)
#: a candidate's calibration score must land within. Lower edge is :data:`C_MIN`.
CALIBRATION_BAND: tuple[float, float] = (0.75, 1.0)

#: Minimum meaningful drop (success-rate units) — the pre-registered effect
#: size the decision rule gates on. Fixed against the archived power
#: simulation (spikes/results/cocarry/rung3/cocarry_rung3_power_sim.json).
DELTA_MIN: float = 0.20

#: Calibration seed-clusters (candidate + cooperative reference). Disjoint
#: from the Rung-2 selection S (10000-10011) and validation V (20000-20023)
#: and from the measurement set below.
CALIBRATION_SEEDS: tuple[int, ...] = tuple(range(40000, 40012))

#: Measurement seed-clusters (reference + each shifted teammate; paired —
#: the same seeds across all conditions so Δ is a within-seed paired contrast).
#: 12 clusters (>= the ADR-026 §Validation 10-12 forward-statistics floor).
#: Disjoint from S, V, and the calibration set.
MEASUREMENT_SEEDS: tuple[int, ...] = tuple(range(30000, 30012))

#: One-sided confidence level for the CI on Δ (the decision rule reads the
#: 95% lower confidence bound).
CI_ALPHA: float = 0.05

#: Cluster-bootstrap resample count.
N_BOOTSTRAP: int = 10000

#: Deterministic substream name for the bootstrap RNG (P6 / ADR-002 — never
#: an ad-hoc np.random call).
_BOOTSTRAP_SUBSTREAM: str = "cocarry.rung3.bootstrap"

#: Pre-committed verdict labels.
VERDICT_DROP: str = "ph_reduces_cooperation"
VERDICT_NULL: str = "thesis_disconfirming_null"
VERDICT_INDETERMINATE: str = "indeterminate"

#: The pre-committed null rule (R-2026-06-B §15 Rung 3). Recorded verbatim so
#: the verdict text cannot be re-narrated post hoc.
NULL_RULE: str = (
    "The coupling positive-control passes (single-arm ~ 0, established at "
    "Rung 1) AND Δ ~ 0 (the one-sided 95% CI excludes Δ_min) ⇒ a "
    "thesis-disconfirming null: no policy-heterogeneity penalty on this axis "
    "under this task. Reported as such — NEVER re-described as a coupling "
    "failure (the coupling is already established)."
)

#: The mandatory caveat when the calibration gate excluded any candidate.
NULL_CAVEAT: str = (
    "The capability gate truncates heterogeneity from above (it excludes "
    "teammates the cooperative reference cannot succeed with), which biases Δ "
    "toward zero. When exclusions occurred, a Δ≈0 result may NOT be read as "
    "'the PH axis does not couple to cooperation' — only as 'among "
    "capability-matched policies, no qualifying drop was observed'."
)


# ===========================================================================
# Pure statistics (Tier-1; no SAPIEN). The inference the pre-registration
# locks. Operate on plain {seed: bool} maps so the Tier-1 tests can pin them.
# ===========================================================================


def success_rate(per_seed_success: Mapping[int, bool]) -> float:
    """Joint-success rate over a ``{seed: success}`` map (ADR-026 §Decision 4; empty ⇒ nan)."""
    if not per_seed_success:
        return float("nan")
    return float(np.mean([bool(v) for v in per_seed_success.values()]))


def passes_capability_gate(calib_success_rate: float, *, c_min: float = C_MIN) -> bool:
    """Whether a candidate clears the capability gate (ADR-026 §D4; R-2026-06-B §15)."""
    return bool(calib_success_rate >= c_min)


def paired_delta(reference: Mapping[int, bool], shifted: Mapping[int, bool]) -> dict[str, Any]:
    """Per-seed + mean paired Δ, reference - shifted (ADR-026 §D4; R-2026-06-B §15).

    The two maps must share the same seed set (the measurement is paired —
    the same goal jitter per seed across reference and shifted), else a
    ``ValueError`` flags the wiring bug loudly.
    """
    if set(reference) != set(shifted):
        msg = (
            "paired_delta: reference and shifted seed sets differ "
            f"({sorted(reference)} vs {sorted(shifted)}); Δ must be a paired "
            "within-seed contrast (R-2026-06-B §15 Rung 3)."
        )
        raise ValueError(msg)
    seeds = sorted(reference)
    per_seed = {s: float(bool(reference[s])) - float(bool(shifted[s])) for s in seeds}
    return {
        "mean_delta": float(np.mean(list(per_seed.values()))) if seeds else float("nan"),
        "per_seed_delta": per_seed,
        "reference_rate": success_rate(reference),
        "shifted_rate": success_rate(shifted),
    }


def trimmed_mean(values: list[float], *, trim: float = 0.25) -> float:
    """Symmetric trimmed mean (the IQM when ``trim=0.25``; ADR-026 §Decision 4); empty ⇒ nan."""
    if not values:
        return float("nan")
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = arr.size
    k = int(np.floor(n * trim))
    core = arr[k : n - k] if n - 2 * k > 0 else arr
    return float(np.mean(core))


def iqm(values: list[float]) -> float:
    """Interquartile mean (25%-trimmed mean) — the secondary, non-gate estimator (ADR-026 §D4)."""
    return trimmed_mean(values, trim=0.25)


def _pooled_delta_point(
    reference: Mapping[int, bool],
    shifted_by_teammate: Mapping[str, Mapping[int, bool]],
    seeds: list[int],
) -> float:
    """Pooled Δ on a (possibly resampled) seed list: ref_rate - mean_teammate(shifted_rate).

    Pooling = the mean over teammates of each teammate's success rate on the
    seed list, subtracted from the reference rate on the same list. This
    weights each teammate equally (not each episode), so a teammate with more
    episodes cannot dominate the pooled estimate (here every teammate has the
    same seed set, so it also equals the grand episode mean).
    """
    ref_rate = float(np.mean([float(bool(reference[s])) for s in seeds]))
    teammate_rates = [
        float(np.mean([float(bool(shifted[s])) for s in seeds]))
        for shifted in shifted_by_teammate.values()
    ]
    return ref_rate - float(np.mean(teammate_rates))


def cluster_bootstrap_delta(
    reference: Mapping[int, bool],
    shifted_by_teammate: Mapping[str, Mapping[int, bool]],
    *,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = CI_ALPHA,
    root_seed: int = 0,
) -> dict[str, Any]:
    """Pooled + per-teammate Δ with a cluster (seed) bootstrap one-sided CI (R-2026-06-B §15).

    The resample unit is the **seed-cluster** (the independent unit — one
    deterministic episode per seed; the only within-seed variation is the
    goal jitter). Each bootstrap iteration resamples seeds with replacement
    and recomputes the pooled Δ on the *same* resampled seeds across the
    reference and every teammate (preserving the paired/clustered structure).
    Returns the pooled point Δ, the one-sided ``1-alpha`` **lower** confidence
    bound (the decision rule reads ``ci_lower > 0``) and the two-sided
    interval for context, plus per-teammate point Δ + lower bounds.

    RNG routes through :func:`concerto.training.seeding.derive_substream`
    (P6 / ADR-002) so the CI is byte-reproducible.
    """
    from concerto.training.seeding import derive_substream

    seeds = sorted(reference)
    for name, shifted in shifted_by_teammate.items():
        if set(shifted) != set(seeds):
            msg = (
                f"cluster_bootstrap_delta: teammate {name!r} seed set differs from "
                "the reference; Δ must be paired within seed (R-2026-06-B §15)."
            )
            raise ValueError(msg)
    rng = derive_substream(_BOOTSTRAP_SUBSTREAM, root_seed=root_seed).default_rng()
    n = len(seeds)
    seed_arr = np.asarray(seeds)

    point = _pooled_delta_point(reference, shifted_by_teammate, seeds)
    boot_pooled = np.empty(n_boot, dtype=np.float64)
    per_teammate_boot: dict[str, NDArray[np.float64]] = {
        name: np.empty(n_boot, dtype=np.float64) for name in shifted_by_teammate
    }
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        resampled = [int(seed_arr[i]) for i in idx]
        boot_pooled[b] = _pooled_delta_point(reference, shifted_by_teammate, resampled)
        for name, shifted in shifted_by_teammate.items():
            ref_rate = float(np.mean([float(bool(reference[s])) for s in resampled]))
            sh_rate = float(np.mean([float(bool(shifted[s])) for s in resampled]))
            per_teammate_boot[name][b] = ref_rate - sh_rate

    lo_q = 100.0 * alpha
    pooled_ci_lower = float(np.percentile(boot_pooled, lo_q))
    pooled_ci = (
        float(np.percentile(boot_pooled, 100.0 * (alpha / 2.0))),
        float(np.percentile(boot_pooled, 100.0 * (1.0 - alpha / 2.0))),
    )
    per_teammate = {}
    for name, shifted in shifted_by_teammate.items():
        pd = paired_delta(reference, shifted)
        per_teammate[name] = {
            "mean_delta": pd["mean_delta"],
            "reference_rate": pd["reference_rate"],
            "shifted_rate": pd["shifted_rate"],
            "ci_lower_one_sided": float(np.percentile(per_teammate_boot[name], lo_q)),
        }
    return {
        "pooled_mean_delta": point,
        "pooled_ci_lower_one_sided": pooled_ci_lower,
        "pooled_ci_two_sided": pooled_ci,
        "n_seed_clusters": n,
        "n_boot": n_boot,
        "alpha": alpha,
        "per_teammate": per_teammate,
    }


def decide(
    *,
    pooled_mean_delta: float,
    pooled_ci_lower_one_sided: float,
    pooled_ci_upper_for_null: float,
    delta_min: float = DELTA_MIN,
    positive_control_holds: bool,
    any_excluded: bool,
) -> dict[str, Any]:
    """Apply the pre-committed decision + null rule (ADR-026 §Decision 4; R-2026-06-B §15 Rung 3).

    Primary rule (drop): conclude PH reduces cooperation iff the pooled mean
    Δ >= ``delta_min`` AND the one-sided 95% CI excludes 0 (``ci_lower > 0``).
    Null rule: the positive-control holds AND Δ≈0 (the CI excludes ``delta_min``
    — here ``pooled_ci_upper_for_null < delta_min``) ⇒ a thesis-disconfirming
    null. Anything else is indeterminate (decision-relevant: collect more
    seed-clusters or sharpen the teammate set).

    ``pooled_ci_upper_for_null`` is the two-sided upper bound used to test
    "CI excludes Δ_min" for the null; the primary drop rule reads the
    one-sided lower bound.
    """
    drop = bool(pooled_mean_delta >= delta_min and pooled_ci_lower_one_sided > 0.0)
    null = bool(positive_control_holds and not drop and pooled_ci_upper_for_null < delta_min)
    if drop:
        verdict = VERDICT_DROP
    elif null:
        verdict = VERDICT_NULL
    else:
        verdict = VERDICT_INDETERMINATE
    out: dict[str, Any] = {
        "verdict": verdict,
        "drop_rule_met": drop,
        "null_rule_met": null,
        "delta_min": delta_min,
        "pooled_mean_delta": pooled_mean_delta,
        "pooled_ci_lower_one_sided": pooled_ci_lower_one_sided,
        "positive_control_holds": positive_control_holds,
    }
    if any_excluded:
        out["null_caveat"] = NULL_CAVEAT
    return out


def cluster_robust_glmm(
    reference: Mapping[int, bool],
    shifted_by_teammate: Mapping[str, Mapping[int, bool]],
) -> dict[str, Any]:
    """Cluster-robust binomial GLM confirmatory on the outcomes (ADR-026 §D4; R-2026-06-B §15).

    Fits ``success ~ shifted`` (1 = a shifted teammate episode, 0 = a matched
    reference episode) as a binomial GLM with **cluster-robust (sandwich)
    inference clustered by seed** via :class:`statsmodels.api.GEE` (binomial
    family, exchangeable working correlation, groups = seed) — the
    cluster-robust binomial-GLMM confirmatory the pre-registration names. The
    reference rows enter once; each teammate contributes one row per seed, so
    the ``shifted`` coefficient is the pooled log-odds contrast and the
    grouping clusters the reference + teammate rows that share a seed. A
    significantly **negative** coefficient confirms the bootstrap drop;
    non-significant confirms the null.

    Returns the coefficient (log-odds), its cluster-robust Wald p-value and
    95% CI, and the implied odds ratio. Degenerate data (all-success or
    all-failure, perfect separation) is reported as ``status='degenerate'``
    rather than raising — the bootstrap remains the headline, the GLMM is
    confirmatory.
    """
    # pandas + statsmodels are OPTIONAL (the `eval` extra + statsmodels) — the
    # GEE is the confirmatory, NOT the gate (the cluster bootstrap is the
    # pre-registered headline). Imported dynamically so the base install (and
    # the static type-check) does not require the heavy stats stack; absent ⇒
    # the confirmatory is skipped, the verdict is unaffected (R-2026-06-B §15).
    import importlib

    try:
        pd = importlib.import_module("pandas")
        sm = importlib.import_module("statsmodels.api")
        smf = importlib.import_module("statsmodels.formula.api")
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "pandas + statsmodels not installed (optional 'eval' extra + "
            "statsmodels); the GEE confirmatory is skipped. The cluster bootstrap is the "
            "pre-registered headline, so the verdict is unaffected.",
        }

    rows: list[dict[str, Any]] = []
    for s, v in reference.items():
        rows.append({"success": int(bool(v)), "shifted": 0, "seed": int(s)})
    for shifted in shifted_by_teammate.values():
        for s, v in shifted.items():
            rows.append({"success": int(bool(v)), "shifted": 1, "seed": int(s)})
    df = pd.DataFrame(rows)
    # Degenerate guards: the GLM needs >= 2 distinct outcome values AND both
    # the reference and a shifted condition present, else it is not identifiable.
    min_distinct = 2
    if df["success"].nunique() < min_distinct or df["shifted"].nunique() < min_distinct:
        return {
            "status": "degenerate",
            "reason": "no outcome variation or no reference/shifted contrast; "
            "GLM not identifiable (the bootstrap remains the headline)",
            "n_rows": len(df),
        }
    try:
        model = smf.gee(
            "success ~ shifted",
            groups="seed",
            data=df,
            family=sm.families.Binomial(),
            cov_struct=sm.cov_struct.Exchangeable(),
        )
        res = model.fit()
        if res is None:
            return {"status": "error", "reason": "GEE fit returned no result", "n_rows": len(df)}
        coef = float(res.params["shifted"])
        pval = float(res.pvalues["shifted"])
        if not np.isfinite(coef) or not np.isfinite(pval):
            # Near-perfect separation: the coefficient diverges and the Wald
            # p-value is undefined. The bootstrap remains the headline.
            return {
                "status": "degenerate",
                "reason": "near-perfect separation (coefficient diverges, Wald p undefined); "
                "the bootstrap remains the headline",
                "coef_shifted_logodds": coef,
                "n_rows": len(df),
            }
        ci = res.conf_int().loc["shifted"]
        return {
            "status": "ok",
            "model": "binomial GEE (exchangeable, clustered by seed) — cluster-robust",
            "coef_shifted_logodds": coef,
            "p_value": pval,
            "ci95_logodds": [float(ci[0]), float(ci[1])],
            "odds_ratio": float(np.exp(coef)),
            "n_rows": len(df),
            "n_clusters": int(df["seed"].nunique()),
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "n_rows": len(df)}


# ===========================================================================
# Rollout (Tier-2; SAPIEN-gated).
# ===========================================================================


@dataclass(frozen=True)
class ConjunctMetrics:
    """Per-episode co-carry outcome + the success-conjunct breakdown (ADR-026 §D4; R-2026-06-B §15).

    Extends :class:`chamber.benchmarks.cocarry_runner.EpisodeMetrics` with the
    per-conjunct booleans the success predicate ANDs, so a drop's mechanism is
    visible: which of placed / level / unstressed / static tips a shifted
    teammate over (the reference lands at centroid ~0.099 vs the 0.10 radius,
    so transport precision is the likeliest channel — surfaced, not assumed).
    """

    seed: int
    success: bool
    is_placed: bool
    is_level: bool
    is_unstressed: bool
    both_static: bool
    centroid_to_goal: float
    max_tilt_deg: float
    max_stress_proxy: float
    n_steps: int


def _info_bool(info: Mapping[str, Any], key: str) -> bool:
    """Coerce an env-0 info flag (torch/np scalar or (1,)-tensor) to a Python bool."""
    val = info.get(key)
    if val is None:
        return False
    return bool(np.asarray(val.detach().cpu() if hasattr(val, "detach") else val).reshape(-1)[0])


def rollout_pair(
    *,
    env: gym.Env[Any, Any],
    ego_act: Callable[[Mapping[str, Any]], NDArray[np.float32]],
    partner: FrozenPartner,
    seed: int,
    episode_length: int,
) -> ConjunctMetrics:
    """Roll one matched-condition episode: ego seat = ``ego_act``, partner seat = ``partner``.

    The single generic rollout for all three Rung-3 conditions — the matched
    reference / the shifted measurement (ego = frozen incumbent closure) and
    the capability calibration (ego = cooperative reference closure). Captures
    the final-step conjunct breakdown from the env ``info`` (ADR-026 §D4).
    """
    ego_uid = env.get_wrapper_attr("ego_uid")
    partner_uid = env.get_wrapper_attr("partner_uid")
    obs, _ = env.reset(seed=seed)
    partner.reset(seed=seed)
    info: dict[str, Any] = {}
    for _ in range(episode_length):
        action = {ego_uid: ego_act(obs), partner_uid: partner.act(obs)}
        obs, _, terminated, truncated, info = env.step(action)
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break
    tel = env.get_wrapper_attr("get_telemetry")()
    return ConjunctMetrics(
        seed=seed,
        success=_info_bool(info, "success"),
        is_placed=_info_bool(info, "is_placed"),
        is_level=_info_bool(info, "is_level"),
        is_unstressed=_info_bool(info, "is_unstressed"),
        both_static=_info_bool(info, "both_static"),
        centroid_to_goal=_to_float(tel["centroid_to_goal"]),
        max_tilt_deg=_to_float(tel["max_tilt_deg"]),
        max_stress_proxy=_to_float(tel["max_stress_proxy"]),
        n_steps=episode_length,
    )


def build_partner_seat(class_name: str, *, seed: int = 0) -> FrozenPartner:
    """Build a teammate (or the matched partner) on the partner seat (ADR-026 §D4; ADR-009).

    Every teammate is built from the env's single-source-of-truth partner-seat
    geometry (:func:`chamber.envs.cocarry.cocarry_matched_controller_specs`'s
    ``panda_partner`` extras), so the policy is the only thing that varies
    across the matched partner and the policy-shift candidates.
    """
    extra = cocarry_matched_controller_specs()["panda_partner"]
    return load_partner(PartnerSpec(class_name, seed, None, None, extra))


def build_cooperative_ego(*, seed: int = 0) -> FrozenPartner:
    """Build the cooperative-reference ego (matched impedance) for the gate (ADR-026 §D4; ADR-009).

    NOT the frozen incumbent (R-2026-06-B §15 Rung 3 capability-gate rule):
    the matched impedance controller on the ego seat (``panda_wristcam``) is a
    hand-written proven cooperative controller, so pairing a candidate with it
    measures the candidate's capability, defusing the weaker-teammate confound.
    """
    extra = cocarry_matched_controller_specs()["panda_wristcam"]
    return load_partner(PartnerSpec(COOPERATIVE_REFERENCE_EGO_CLASS, seed, None, None, extra))


def evaluate_calibration(
    *,
    candidate_class: str,
    seeds: list[int],
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    render_backend: str | None = None,
) -> list[ConjunctMetrics]:
    """Calibrate a candidate: pair it (partner seat) with the cooperative ego (ADR-026 §D4).

    For each seed: build the matched-condition training env, the cooperative
    reference ego, and the candidate on the partner seat; roll one episode.
    The resulting :func:`success_rate` is gated against :data:`C_MIN`. A fresh
    env per seed mirrors the Rung-0/1/2 runners.
    """
    from chamber.envs.cocarry_obs import make_cocarry_training_env

    metrics: list[ConjunctMetrics] = []
    for s in seeds:
        env = make_cocarry_training_env(
            condition_id="cocarry_matched_panda_pair",
            episode_length=episode_length,
            root_seed=s,
            render_backend=render_backend,
        )
        try:
            coop_ego = build_cooperative_ego(seed=s)
            candidate = build_partner_seat(candidate_class, seed=s)
            coop_ego.reset(seed=s)
            metrics.append(
                rollout_pair(
                    env=env,
                    ego_act=lambda obs, _e=coop_ego: np.asarray(_e.act(obs), dtype=np.float32),
                    partner=candidate,
                    seed=s,
                    episode_length=episode_length,
                )
            )
        finally:
            env.close()
    return metrics


def evaluate_incumbent_vs_partner(
    *,
    cfg: EgoAHTConfig,
    checkpoint_uri: str,
    artifacts_root: Path,
    partner_class: str,
    seeds: list[int],
    episode_length: int = COCARRY_DEFAULT_EPISODE_LENGTH,
    render_backend: str | None = None,
) -> list[ConjunctMetrics]:
    """Evaluate the frozen incumbent (ego) against a teammate (partner seat) (ADR-026 §D4).

    The reference condition uses ``partner_class='cocarry_impedance'``; the
    shifted conditions use a calibrated policy-shift teammate. The ego seat is
    always the SHA-verified frozen incumbent (never retrained). A fresh env +
    frozen-incumbent load per seed mirrors
    :func:`chamber.benchmarks.cocarry_incumbent.evaluate_incumbent_matched`.
    """
    from chamber.benchmarks.cocarry_incumbent import load_frozen_incumbent
    from chamber.envs.cocarry_obs import make_cocarry_training_env

    metrics: list[ConjunctMetrics] = []
    for s in seeds:
        env = make_cocarry_training_env(
            condition_id="cocarry_matched_panda_pair",
            episode_length=episode_length,
            root_seed=s,
            render_backend=render_backend,
        )
        try:
            partner = build_partner_seat(partner_class, seed=s)
            ego_act = load_frozen_incumbent(
                cfg=cfg,
                env=env,
                partner=partner,
                checkpoint_uri=checkpoint_uri,
                artifacts_root=artifacts_root,
            )
            metrics.append(
                rollout_pair(
                    env=env,
                    ego_act=ego_act,
                    partner=partner,
                    seed=s,
                    episode_length=episode_length,
                )
            )
        finally:
            env.close()
    return metrics


def conjunct_failure_summary(metrics: list[ConjunctMetrics]) -> dict[str, Any]:
    """Per-conjunct failure counts over a teammate's episodes (ADR-026 §D4; R-2026-06-B §15 Rung 3).

    Surfaces which conjunct (placed / level / unstressed / static) fails on
    drops so the cooperation-channel the shifted partner tips over is visible.
    """
    if not metrics:
        return {}
    n = len(metrics)
    return {
        "n": n,
        "success_rate": float(np.mean([m.success for m in metrics])),
        "fail_placed": int(sum(1 for m in metrics if not m.is_placed)),
        "fail_level": int(sum(1 for m in metrics if not m.is_level)),
        "fail_unstressed": int(sum(1 for m in metrics if not m.is_unstressed)),
        "fail_static": int(sum(1 for m in metrics if not m.both_static)),
        "centroid_to_goal_p50": float(np.percentile([m.centroid_to_goal for m in metrics], 50)),
        "centroid_to_goal_max": float(np.max([m.centroid_to_goal for m in metrics])),
        "max_tilt_p90": float(np.percentile([m.max_tilt_deg for m in metrics], 90)),
        "stress_p90": float(np.percentile([m.max_stress_proxy for m in metrics], 90)),
    }


__all__ = [
    "CALIBRATION_BAND",
    "CALIBRATION_SEEDS",
    "CI_ALPHA",
    "COOPERATIVE_REFERENCE_EGO_CLASS",
    "C_MIN",
    "C_MIN_DERIVATION",
    "DELTA_MIN",
    "MATCHED_PARTNER_CLASS",
    "MEASUREMENT_SEEDS",
    "NULL_CAVEAT",
    "NULL_RULE",
    "N_BOOTSTRAP",
    "VERDICT_DROP",
    "VERDICT_INDETERMINATE",
    "VERDICT_NULL",
    "ConjunctMetrics",
    "build_cooperative_ego",
    "build_partner_seat",
    "cluster_bootstrap_delta",
    "cluster_robust_glmm",
    "conjunct_failure_summary",
    "decide",
    "evaluate_calibration",
    "evaluate_incumbent_vs_partner",
    "iqm",
    "paired_delta",
    "passes_capability_gate",
    "rollout_pair",
    "success_rate",
    "trimmed_mean",
]
