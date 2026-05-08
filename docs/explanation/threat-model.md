# Threat model

The Phase-0 threat surface is intentionally small: no network listeners,
no user-input parsing at runtime, no database. Two named threats:

## T1 — Malicious dependency upgrade

A compromised package in the dependency graph injects malicious code.

**Mitigations:**

- `uv.lock` pins every transitive dependency by content hash.
- `security.yml` runs OSV-Scanner on every PR and nightly.
- dependabot opens batched upgrade PRs; CI runs the full suite on each.
- All GitHub Actions steps are pinned by SHA (OSSF Scorecard requirement).

## T2 — Tampered checkpoint file

A manipulated partner-zoo checkpoint produces unsafe behaviour without
triggering the CBF safety filter.

**Mitigations:**

- `concerto.training.checkpoints` (Phase 1) maintains a SHA-256
  manifest for every checkpoint in the partner zoo.
- The `FrozenPartner` shield (`chamber.partners.interface`) prevents
  runtime modification of a loaded checkpoint's attributes.
- ADR-009 §Validation criteria requires the MEP entropy filter to verify
  partner diversity at zoo-construction time (Phase 1).

## Out of scope for Phase 0

- Network-level adversarial attacks on the communication channel.
  (URLLC degradation models benign packet loss; adversarial injection
  is a Phase-3 concern.)
- Side-channel attacks on the CBF-QP solver.
- Poisoning of the training dataset (no dataset in Phase 0; offline RL
  from sim only).
