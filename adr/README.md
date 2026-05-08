# ADR snapshot

This directory is a frozen snapshot of the canonical Architecture Decision
Records (ADRs) maintained in the project's planning kit
(`phase0_reading_kit/adrs/`).

It exists so contributors don't have to clone the planning kit to read the
project's design decisions. The snapshot is the authoritative reference for
*this commit* of the codebase.

## Reading order

1. [`ADR-INDEX.md`](ADR-INDEX.md) — status board and locking rule.
2. ADR-001 through ADR-015 — individual decisions, in numeric order.

## Sync policy

The snapshot is updated only via PR. Each sync PR's description links the
planning-kit commit SHA the snapshot was taken from. The two sources must
agree at every release tag.

If you spot a discrepancy between this snapshot and a downstream module's
behaviour, open an issue with the `adr` label — do not edit either copy
directly. New ADRs are drafted in the planning kit first and then synced
here.
