# SPDX-License-Identifier: Apache-2.0
"""CONCERTO training stack — ego-AHT HAPPO wrapper + seeding + logging.

Implements the frozen-partner ego-AHT training algorithm from ADR-002
using the HARL fork (ADR-002 §"HARL dependency"). The seeding module
provides the determinism harness required by project principle P6.
"""
