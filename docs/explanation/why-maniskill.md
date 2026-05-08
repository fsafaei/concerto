# Why fork above ManiSkill v3?

ManiSkill v3 (SAPIEN 3 + Warp-MPM) is the only publicly available simulator
that simultaneously supports: heterogeneous robot embodiments, contact-rich
manipulation, GPU-parallel rendering, and a Gymnasium-compatible API.

The design decision (ADR-001) is to wrap — never fork — ManiSkill v3.
The three wrapper layers (`PerAgentActionRepeatWrapper`,
`TextureFilterObsWrapper`, `CommShapingWrapper`) add CHAMBER-specific
behaviour without modifying the upstream package. This preserves the ability
to track upstream security and bug fixes, and avoids the supply-chain risk of
a private fork diverging silently.

The wrapper strategy is enforced mechanically: `tests/unit/test_no_private_imports.py`
fails if any `src/chamber/` module imports a `_private` symbol from
`mani_skill`.
