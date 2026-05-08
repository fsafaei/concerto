# Tutorial: Run the Stage-0 smoke test

This tutorial walks through the Stage-0 acceptance script defined in
[ADR-001 §Validation criteria](../reference/adrs.md).

The smoke test validates that the three CHAMBER wrapper layers compose
correctly on top of ManiSkill v3 with three heterogeneous robot UIDs.
It does **not** test any task logic — its only job is to confirm the
infrastructure (wrapper chain, observation namespacing, comm injection,
per-agent action repeat) works as designed.

---

## What you need

| Requirement | Notes |
|-------------|-------|
| Python 3.11 or 3.12 | Managed by `uv` |
| `uv sync` completed | Installs all dependencies including `mani-skill==3.0.1` |
| **Vulkan-capable GPU** | Required by SAPIEN (ManiSkill's physics backend) for Tier-2 tests |

**Without a GPU:** Tier-1 (wrapper-structure) tests always run and cover
ADR-001 conditions (a), (b), (c) using a lightweight fake env. Tier-2
(real ManiSkill) tests are automatically skipped.

---

## Run it

```bash
bash scripts/repro/stage0_smoke.sh
```

Or equivalently:

```bash
uv run pytest -m smoke -x -v -k stage0
```

A successful run on a GPU machine prints:

```
PASSED tests/integration/test_stage0_smoke.py::TestADR001CondAFake::test_pass_a_three_agent_namespaced_obs
PASSED tests/integration/test_stage0_smoke.py::TestADR001CondBFake::test_pass_b_comm_channel_present
PASSED tests/integration/test_stage0_smoke.py::TestADR001CondCFake::test_pass_c_slow_agent_action_update_interval
PASSED tests/integration/test_stage0_smoke.py::TestRealManiSkillSmoke::test_make_stage0_env_returns_gymnasium_env
PASSED tests/integration/test_stage0_smoke.py::TestRealManiSkillSmoke::test_100_steps_no_error
```

On a CPU-only machine the last two are `SKIPPED` — this is expected.

---

## How it works

### The wrapper chain

`make_stage0_env()` (in `chamber.benchmarks.stage0_smoke`) builds the
following wrapper stack (innermost → outermost):

```python
# 1. Inner env — ManiSkill v3 custom task with 3-robot UIDs
env = _Stage0SmokeEnv(
    robot_uids=("panda_wristcam", "fetch", "allegro_hand_right"),
    num_envs=1,
    obs_mode="state",
)

# 2. Per-agent action repeat (ADR-001 §Decision item 2)
env = PerAgentActionRepeatWrapper(env, action_repeat={
    "panda_wristcam": 5,      # 100 Hz / 5 = 20 Hz
    "fetch": 10,              # 100 Hz / 10 = 10 Hz
    "allegro_hand_right": 2,  # 100 Hz / 2 = 50 Hz
})

# 3. Texture / obs-channel filter (ADR-001 §Decision item 1)
env = TextureFilterObsWrapper(env, keep_per_agent={
    "panda_wristcam":      ["rgb", "depth", "joint_pos", "joint_vel"],
    "fetch":               ["state", "joint_pos"],
    "allegro_hand_right":  ["joint_pos", "joint_vel", "tactile"],
})

# 4. Comm shaping — outermost, closest to the agent (ADR-001 §Decision item 3)
env = CommShapingWrapper(
    env,
    channel=FixedFormatCommChannel(latency_ms=50.0, drop_rate=0.05),
)
```

### ADR-001 pass conditions

| Condition | What is checked | Where |
|-----------|-----------------|-------|
| **(a)** Three independently-namespaced proprio dicts | `obs["agent"]` contains three separate dicts, one per uid | `TestADR001CondAFake` |
| **(b)** Shaped comm channel present | `obs["comm"]` exists after `reset()` and `step()` | `TestADR001CondBFake` |
| **(c)** Slow agent action interval matches 1/rate | Over 100 steps, panda updates exactly 20 times (repeat=5) | `TestADR001CondCFake` |

### Wrapper semantics in brief

**`PerAgentActionRepeatWrapper`** — holds each slow agent's most recent
action for `repeat` env ticks, re-submitting it instead of the newly
submitted action. The env runs at its native frequency; only the effective
*action update rate* is reduced.

**`TextureFilterObsWrapper`** — zero-masks observation channels that are
not in the agent's `keep` set. Shapes are preserved so vectorised rollout
works across heterogeneous agents with different modality budgets.

**`CommShapingWrapper`** — adds `obs["comm"]` containing the output of
`channel.encode(obs)`. In M1 the stub channel returns `{}`;
M2 fills it with fixed-format pose / task-state predicate / AoI packets.

---

## If a condition fails

If the real-ManiSkill tests fail with a wrapper error (not a Vulkan error),
**do not patch around it**. The test output will print which ADR-001
condition failed. Open an issue tagged `adr-retrigger` against ADR-001 and
ADR-005 before making any code changes. See the failure-mode rule in the
project's contributor guide.
