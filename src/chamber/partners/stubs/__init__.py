# SPDX-License-Identifier: Apache-2.0
"""Phase-1 partner stubs (ADR-010 §Decision).

This sub-package fixes the :class:`~chamber.partners.api.FrozenPartner`
interface for the OpenVLA + CrossFormer Phase-1 work so that landing the
real implementations is mechanical: drop in the inference harness and
replace the :class:`NotImplementedError`s. Phase-0 only ships the stubs.
"""
