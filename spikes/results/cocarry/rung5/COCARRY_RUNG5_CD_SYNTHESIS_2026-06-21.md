# Co-carry Rung-5 (co-design axis) — Stage-7 synthesis + the decision now due

**Status.** Decision input for the founder and the review board. This document
synthesises the Rung-5 co-design (CD) measurement; it states the question the
result forces and lays out the options, but it makes **no scientific or strategic
call** — that is explicitly a founder/board call, not a mechanical next run
(ADR-026 §Decision 4; R-2026-06-B §15; R-RES-01).

**Provenance.** Branch `feat/cocarry-rung5-codesign` (Stage-0 commit `221e696`
through `abd968d`, pushed; **not merged**). Pre-registration tag
`prereg-cocarry-rung5-codesign-2026-06-20` (locked before any measurement).
Footnote: the committed prereg JSON's internal `status` string still reads
"DRAFT" — cosmetic only; the git tag + commit are the lock.

---

## 1. What Rung-5 set out to do

The prior ladder (Rungs 2–4e) closed with **PH and EH both benign under fair
matching**, but every headline on the compliant-coupling task used a *hand-written
cooperative-impedance stand-in* on the ego seat, because the rigid-trained Rung-2
incumbent did not survive the coupling change. Rung-5 was the definitive run: (1)
re-freeze a genuinely **trained** incumbent on the fixed task, and (2) measure it
against two partner sets differing in **exactly one factor — co-design**:

- **Arm A (control):** capability-matched, co-designed control-style partners
  (stiff / admittance / nullspace impedance) — expected null.
- **Arm B (decisive):** independently-competent but **non-co-designed** partners
  (selfish_goal / selfish_effort) — the regime the thesis is about.

The in-kit motivation was the Rung-2 STOP (854 N internal stress against a frozen
non-cooperating partner): the non-co-designed regime is the one we could not
co-optimise, and the hypothesis was that *that* is where heterogeneity bites.

---

## 2. The run, end to end

| Stage | What | Outcome |
|---|---|---|
| 0 | Pre-register (tagged) + env ceiling-consistency fix + Arm-B partners + power sizing | n=**28** (equivalence-null adequacy); reward+predicate aligned to the coupling instrument at f_max 365.6 |
| 1 | Rig re-validation | PASS — invariance 0.61 N, matched 12/12, success-stress p99 **288.7 N in band [234,318]**, single-arm 0/6 (coupling real) |
| 2 | Capability calibration (vs cooperative reference, C_min=0.75) | Arm A admitted stiff/admittance/nullspace (**slew excluded**); Arm B admitted selfish_goal/selfish_effort (**selfish_station excluded**) → ≥2 ⇒ PROCEED. Roster **flipped vs Rung-3** (slew out, nullspace in) — confirms the gate is substrate-specific |
| 3 | Re-freeze (heavy) | From-scratch **STOPPED** at the pre-registered 300k early-gate (transport-but-tilted: tilt 34–82° vs 15°; leveling reward saturates). Locked escalation → **residual-on-impedance**: **FROZEN** step 150k, held-out V **24/24**, SHA `152bdb3f` |
| 4–5 | CD measurement (eval-only, n=28) | **CONFOUNDED_BY_INCUMBENT_BRITTLENESS** (below) |

Every gate fired as pre-registered, and two confounds were caught by *cheaper*
checks before they could mislead: the from-scratch stall (caught at 300k, not
900k), and — decisively — the incumbent-brittleness confound (caught by the
base-robustness control, below).

---

## 3. The result

Mechanically, the measurement looks like the thesis landing:

- Reference (incumbent + matched) reconfirms **1.000**.
- **Arm B** pooled Δ = **+0.75**, replicated **2/2** (selfish_effort 0.00,
  selfish_goal 0.50) → `ph_reduces_cooperation`.
- **Arm A** pooled Δ = +0.155 (indeterminate) — but **not a clean null**: the
  *co-designed* `admittance` partner drops it to 0.536 (over-tilts to 17°), while
  stiff/nullspace hold at 1.000.

The driver's mechanical matrix labelled this "THESIS_LANDS." **It is wrong**, and
the base-robustness control inverts it. Running the **structured base cooperative
ego (no residual)** against every partner on the *same* measurement seeds:

| partner (all capability-matched) | structured **BASE** | residual **INCUMBENT** |
|---|---|---|
| matched / stiff / nullspace | 1.000 | 1.000 |
| **admittance** (co-designed) | **1.000** | 0.536 |
| **selfish_goal** (non-co-designed) | **1.000** | 0.500 |
| **selfish_effort** (non-co-designed) | **1.000** | 0.000 |

`base_robust_all_partners: True`; `n_genuine_drop_arm_b: 0`.

**The structured controller cooperates 100 % with every partner — co-designed and
non-co-designed alike.** Every incumbent "drop" is therefore an artifact of the
*incumbent*, not the partner: verdict **`CONFOUNDED_BY_INCUMBENT_BRITTLENESS`**.

(Integrity note: the base control first read as a partial drop; that was a
stateful-ego reset bug — `rollout_pair`/`rollout_incumbent_episode` reset the
partner but not the ego, and the impedance base is stateful. Fixed in both the
measurement and the Stage-3b diagnostic; the frozen incumbent is a stateless
policy and is unaffected — step/SHA/V validation unchanged.)

---

## 4. The mechanism

The base impedance controller is **partner-agnostic by construction**: it always
drives *its own* bar-end to the cooperative target, so it levels the bar
regardless of what the partner does — passive (admittance), off-objective
(selfish), or active (matched/stiff). It cooperates with all of them.

The residual was trained **only against the active matched partner**. It learned a
correction tuned to that partner's active co-leveling, which **reduces stress with
that partner** (the freeze-time diagnostic: −32 N p90) but **fails** the moment the
partner stops actively co-leveling — admittance (passive) and the selfish partners
all leave the residual's learned correction mis-targeted, and the bar over-tilts.
This is textbook **overfitting to the training partner**, surfaced as
off-distribution brittleness.

Two layers, across the whole re-freeze:
1. **From-scratch RL** could not discover leveling at all (Stage-3 STOP).
2. **Residual-RL** discovered it *with its training partner* but is brittle to
   every other partner.
3. The **structured controller** is the only robustly general cooperator.

---

## 5. Scientific bottom line (what is and isn't established)

- **Established (high confidence):** on this co-carry task, a well-designed
  **structured controller cooperates with every capability-matched partner**,
  co-designed and non-co-designed. The task is **robustly solvable** without any
  trained cooperation.
- **Established:** the **learned incumbent is partner-brittle** (overfits its
  training partner); from-scratch RL is worse (cannot even level).
- **NOT established — and not testable here:** "co-design degrades cooperation"
  (confounded by incumbent brittleness) **and** "co-design is inert" (the task is
  too forgiving to test it). The CD axis is **not isolable on this task with a
  learned incumbent**.

This is consistent with the ladder's PH/EH nulls: under fair matching, this
co-carry task does not expose a cooperation cost — now extended with the reason
*why a learned incumbent looked like it did*: brittleness, not co-design.

---

## 6. The strategic significance (the question this forces)

The result **sharpens the question** the decision has to grapple with. It is no
longer "is co-carry too forgiving for heterogeneity?" It is now:

> **Does trained cooperation add any value over a well-designed structured
> controller — and if so, where?**

Because on co-carry the answer is stark: **the structured controller wins and the
trained residual loses.** That is a direct, evidence-backed challenge to a
"we train the cooperation" framing — produced by our own pre-registered,
adversarially-checked pipeline, which makes it credible rather than dismissible.

It does **not** by itself decide the thesis — co-carry is a forgiving task, and a
forgiving task is exactly where a structured controller *should* win. But it
removes the option of leaning on co-carry as evidence *for* trained cooperation,
and it makes the value-add question first-order.

---

## 7. The decision now due (options, not a call)

The verdict lands **off the two clean branches** of the pre-registered matrix
(it is neither "thesis lands" nor a plain "conclusive negative"). It routes to the
pre-committed **harder-task regime**, *plus* a new first-order constraint:
**incumbent construction** (how to get a robustly-general learned cooperator) is
now the gating problem — a harder task alone will not help while the learned route
is brittle. Framed against the strategy memo
(`01_strategy/COCARRY_NULL_STRATEGIC_DECISION_2026-06-20.md`, not on this host):

- **Option B — reframe around the structured controller.** Treat the structured
  three-layer stack as the deliverable and the co-carry result as evidence that a
  well-designed controller is a robust cooperator where RL is brittle. Strongest
  where the program's value is *safety/robustness*, not *learned* cooperation.
- **Sharper Option A — harder-task regime, framed as a value-add test.** Build a
  contact-rich, asymmetric-role task where the structured controller *struggles*,
  and ask specifically whether trained cooperation **adds value the structured
  controller cannot reach** — trained with partner **diversity** (not a single
  partner) so brittleness is controlled for by construction.

**The diversity-training falsifier belongs inside Option A, not on co-carry.**
On co-carry the base is already robust to all partners, so a diversity-trained
residual could at most *match* the base — it could rule out "intrinsically broken"
but could never show **value-add**, because there is no co-design cost here for it
to overcome. The question that matters needs a task where the base struggles.

---

## 8. Proposed record updates (for review, not yet applied)

- **ADR-026 §Validation criteria / axis status.** Record CD as
  **not-established-on-co-carry (confounded by incumbent brittleness)** — neither
  promoted nor disconfirmed; the construct requirement (coupling-validity) was met
  but the *measuring instrument* (a learned incumbent) was not valid here. Add an
  **Open-question**: a coupling-valid heterogeneity measurement additionally
  requires an incumbent that is **robust to the partner distribution** (else the
  ego's brittleness, not the axis, drives the result) — analogous to the
  capability gate on partners, now applied to the ego.
- **Co-carry spec.** Note the from-scratch leveling-gradient saturation and the
  residual partner-overfitting as task/method findings; record that eval of a
  learned incumbent must carry a **structured-base robustness control** as the
  null for "is a drop the partner or the ego."
- **R-RES-01 (cooperation-proof risk).** Update: the risk is **not** un-armed
  (no clean co-design positive) and **not** cleanly closed (the task can't test
  it); it is **re-pointed** — the open question is value-add of trained
  cooperation over a structured controller, deferred to the harder-task regime.

---

## 9. Committed evidence

| Artifact | Content |
|---|---|
| `cocarry_rung5_codesign_prereg.json` (tag) | locked protocol, n=28, both arms, decision rules |
| `cocarry_rung5_rig_revalidation.json` | Stage-1 PASS |
| `cocarry_rung5_calibration_roster.json` | Stage-2 roster (admitted/excluded) |
| `cocarry_rung5_refreeze_train.json` / `…_residual_train.json` | from-scratch + residual training manifests |
| `cocarry_rung5_freeze_selection.json` | from-scratch 300k STOP (per-checkpoint table) |
| `cocarry_rung5_freeze_manifest.json` | **FROZEN** incumbent v2: step 150k, SHA `152bdb3f`, V 24/24, base-vs-incumbent diagnostic |
| `cocarry_rung5_cd_measurement.json` | Stage 4-5: `CONFOUNDED_BY_INCUMBENT_BRITTLENESS` + the base-robustness control |
| `cocarry_rung5_power_sim.json` | n=28 sizing |

Drivers under `scripts/repro/_cocarry_rung5_*.py`; env/training infra
(`residual_base_ego` wrapper + `EnvConfig` forwarding) byte-identical on the
default path. `make verify`-relevant gates green on touched files (the 2
pre-existing Stage-1 AS failures are issue #215, unrelated).

---

## 10. Standing flags

- **This was the last large axis-measurement spend.** The verdict is the input
  the M7 / M10-narrative / D-046 scope call and the strategy memo were waiting on.
- **Do not merge** — founder/board review. No scientific or strategic call has
  been made in this document or the branch.
- The methodology held under stress: a pre-registered pipeline that lets a *cheap*
  check (the base-robustness control) overturn a *mechanical* positive verdict,
  and reports the correction rather than the headline, is what turned a spurious
  "thesis lands" into a trustworthy, decision-relevant negative.
