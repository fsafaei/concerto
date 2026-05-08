# How-to: Run a spike with a custom hypothesis

!!! note "Phase-0 placeholder"
    The full spike harness lands in M5. The walkthrough below shows the
    M2 comm-degradation surface that the Stage-2 CM spike consumes;
    other axes get their own walkthroughs as they ship.

## Steps (sketch)

1. Copy the nearest existing pre-registration YAML from
   `spikes/preregistration/` as a template.
2. Edit the hypothesis, threshold, and comparison conditions.
3. Commit the YAML and create a git tag (`stage<n>-<axis>-<date>`).
4. Run `chamber-spike run --axis <axis>`.

*(Full content added in M5.)*

## Walkthrough: comm degradation (Stage-2 CM)

The Stage-2 CM spike sweeps the URLLC + 3GPP Release 17 anchored
profiles shipped in `chamber.comm.URLLC_3GPP_R17`. The six rows are
the pre-registered conditions; do not vary them post-launch (per
[ADR-007](../reference/adrs.md) §Stage 2 CM and the project's
pre-registration discipline).

### 1. Inspect the available profiles

```python
from chamber.comm import URLLC_3GPP_R17

for name, profile in URLLC_3GPP_R17.items():
    print(name, profile)
```

The output lists the six rows in order of aggressiveness
(`ideal` → `saturation`). The `saturation` row is held aside for the
QP-saturation property test (it is *expected* to trip the
`ChamberCommQPSaturationWarning`).

### 2. Compose the wrapper stack

```python
from chamber.comm import (
    CommDegradationWrapper,
    FixedFormatCommChannel,
    URLLC_3GPP_R17,
)
from chamber.envs import CommShapingWrapper

# Pick a profile from the pre-registered table; do not edit values.
profile = URLLC_3GPP_R17["factory"]

channel = CommDegradationWrapper(
    FixedFormatCommChannel(),
    profile,
    tick_period_ms=1.0,        # 1 ms per env tick
    root_seed=0,               # determinism (P6)
)

env = CommShapingWrapper(inner_env, channel=channel)
```

### 3. Step the env and read the degraded packet

```python
obs, _ = env.reset(seed=42)
for _ in range(1000):
    obs, *_ = env.step(action)
    packet = obs["comm"]
    # Safety filter (M3) reads packet["pose"][uid] / packet["aoi"][uid].
```

### 4. Pull telemetry for the spike report

The wrapper exposes a `stats` snapshot for the report consumer:

```python
print(channel.stats.dropped, channel.stats.delivered)
print(channel.stats.latency_ticks[:10])
```

The Stage-2 CM spike's three-table emitter feeds these counters into
the per-condition column of the report
([ADR-014](../reference/adrs.md) §Decision).

### 5. Verify the saturation guard

Before launching a long sweep, run the saturation guard locally to
confirm the chosen profile honours the ADR-004 OSCBF target:

```python
from chamber.comm import saturation_guard

saturation_guard(profile)  # silent for ideal/urllc/factory/wifi/lossy
```

The `saturation` row deliberately fires
`ChamberCommQPSaturationWarning` — that is the
[ADR-006](../reference/adrs.md) §Risks R5 contract.
