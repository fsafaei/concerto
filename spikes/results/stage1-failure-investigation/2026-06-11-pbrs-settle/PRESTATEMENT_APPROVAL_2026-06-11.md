# Pre-statement + ADR-note approval record — PBRS settle

**Date.** 2026-06-11.
**Author.** Farhad Safaei.

`PRESTATEMENT.md` and `ADR_NOTE_DRAFT.md` (this directory) are **APPROVED as drafted** by
founder decision of 2026-06-11: gated potential as primary; α pair (0.1 / 0.5) as proposed;
boundary convention as stated. Two riders attach:

1. **Characterization reporting rider.** The pre-flight's 1-of-6 cube-drift nuance means
   settling can cost the place flag at some hold poses. The per-arm instrument adds **place
   retention during stillness**: for cold episodes, report placed∧static joint occupancy per
   step (placed steps, static steps, placed∧static steps, P(static | placed) AND
   P(placed | static)), not only the ever_placed / success scalars, so a drift-on-settle
   failure mode is visible in the record if it occurs.
2. **Phase-4b menu rider.** The ungated variant is NOT run in this slice under any outcome.
   If both α arms fail AND the entry-transient signature is visible in the traces
   (place-entry avoidance, or place regression concentrated at the sphere boundary), the
   ungated variant is listed in the consultation brief as a pre-staged cheap alternative —
   drafted, never launched.

**From the first run's start the pre-statement is FROZEN (I8): no mid-run changes; runs
chained C-lo s0 → s1 → s2 → C-hi s0 → s1 → s2; halt-without-retry.** The shaping lands behind
`shaping.settle_alpha` (default off) in its own `feat` PR with the Rev 18 note; evidence stays
separate per the established pattern. The gate spike does not launch from anything in this
slice.
