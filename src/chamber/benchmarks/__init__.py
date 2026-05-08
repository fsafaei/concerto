# SPDX-License-Identifier: Apache-2.0
"""CHAMBER spike runners — Stage 0/1/2/3 benchmark experiments.

Implements the staged heterogeneity-axis spike protocol from ADR-007.
Each spike module reads a pre-registered YAML (``spikes/preregistration/``)
and refuses to run if the SHA does not match its git tag (project principle P4).
"""
