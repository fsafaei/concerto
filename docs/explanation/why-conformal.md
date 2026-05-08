# Why conformal CBFs?

The exponential CBF-QP backbone (Wang–Ames–Egerstedt 2017) provides
instantaneous safety guarantees but requires tight a-priori bounds on
partner behaviour. In heterogeneous ad-hoc teamwork those bounds are
unknown at deployment time.

The conformal overlay (Huriot–Sibai 2025) relaxes this requirement: it
maintains a running average-loss guarantee (Theorem 3) with a
self-tuning slack variable λ that widens when the partner's actions
are worse than expected and tightens when they are better. The OSCBF
(Morton–Pavone 2025) resolves the multi-arm coupling into tractable
per-arm QPs.

The key open theoretical question is whether the Huriot–Sibai
average-loss bound can be sharpened to a per-step bound. This is not
required for Phase 0 correctness but would strengthen the safety claim.
See ADR-004 §Open questions.

## Communication degradation and the conformal slack

Conformal CBFs only stay tight if the safety filter receives partner
poses *fresh*. When the comm channel degrades — added latency, jitter,
or dropped packets — the filter must reason about *what it does not
know yet*. CHAMBER models this explicitly via the
[`CommDegradationWrapper`](../reference/api.md#chambercomm) that wraps
the mandatory fixed-format channel.

The wrapper is anchored to **URLLC + 3GPP Release 17** industrial-trial
data (see [ADR-006](../reference/adrs.md)) and ships six named
`DegradationProfile` rows under `chamber.comm.URLLC_3GPP_R17`:

| Profile | Latency mean (ms) | Latency std (ms) | Drop rate |
|---------|-------------------|------------------|-----------|
| `ideal` | 0.0 | 0.0 | 0.0 |
| `urllc` | 1.0 | 0.0 | 1e-6 |
| `factory` | 5.0 | 0.1 | 1e-4 |
| `wifi` | 10.0 | 1.0 | 1e-2 |
| `lossy` | 30.0 | 5.0 | 1e-2 |
| `saturation` | 100.0 | 10.0 | 1e-1 |

On every encode, the wrapper:

1. Bernoulli-drops the fresh packet with probability `drop_rate`. On
   drop, the previously visible packet is returned with each per-uid
   AoI bumped by one tick (the receiver knows it is staler now).
2. Otherwise queues the packet with delay drawn from
   `Normal(latency_mean_ms, latency_std_ms)`, clipped to
   `[0, 2 * latency_mean_ms]`. Packets due at the current tick are
   released; the most-recently-sent winner becomes the visible packet,
   with its per-uid AoI bumped by the in-queue dwell time.

The packet itself carries a per-uid **Age of Information** field
(see Ballotta & Talak 2024 for the formal definition). AoI is the
proxy the conformal slack `λ` reads: when the channel degrades, AoI
grows, and the conformal layer widens `λ` to absorb the additional
prediction-error variance.

Two safety properties protect this composition:

- **QP saturation guard.** The wrapper never silently allows a
  configuration that would saturate the inner CBF QP solver beyond the
  ADR-004 OSCBF target (1 ms). A `ChamberCommQPSaturationWarning`
  fires under the `saturation` profile (drop ≥ 10 % or
  latency ≥ 100 ms), per ADR-006 §Risks R5.
- **Determinism.** Every random draw — drop coin, latency sample —
  is derived from `concerto.training.seeding.derive_substream` so two
  resets with the same root seed produce byte-identical degradation
  traces. This keeps the Stage-2 CM spike's pre-registration auditable.

The full M2 wrapper composition that the Stage-0 smoke and Stage-2 CM
spike both use is:

```python
from chamber.comm import (
    CommDegradationWrapper,
    FixedFormatCommChannel,
    URLLC_3GPP_R17,
)

channel = CommDegradationWrapper(
    FixedFormatCommChannel(),
    URLLC_3GPP_R17["factory"],
    tick_period_ms=1.0,
    root_seed=0,
)
```

The fixed-format channel itself ships **no degradation** — encoding
logic stays pure and trivially testable, and degradation is composed
externally (see [ADR-003](../reference/adrs.md) §Decision).
