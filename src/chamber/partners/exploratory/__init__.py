# SPDX-License-Identifier: Apache-2.0
"""EXPLORATORY partner namespace, quarantined from the leaderboard surface (ADR-009; ADR-027).

Partners under this namespace are diagnostic instruments, not zoo
members: they are never registered via
:func:`chamber.partners.registry.register_partner`, never eligible for
leaderboard-facing runs, and each is activated only through an
explicit ``EgoAHTConfig.exploratory`` knob that the gate-facing
:class:`chamber.benchmarks.stage1_common.TrainedPolicyFactory` refuses
at construction (the safety-loud-fail pattern). Promoting an
exploratory partner to the benchmark surface requires an ADR, not an
import.
"""
