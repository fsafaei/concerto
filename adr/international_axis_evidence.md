# International evidence sweep — supplement to ADR-007 (heterogeneity-axis selection)

**Status.** Supplementary evidence brief — does **not** modify ADR-007.
**Author.** Farhad Safaei
**Date.** 2026-05-08
**Tags.** ADR-007, v0.2 §3.2, v0.2 §3.4, axis enumeration

## Purpose

ADR-007's Decision is currently `pending` because two inputs are
unresolved: the Phase-0 axis spikes (≥20pp gap measurements) and the
8-week DACH discovery cohort synthesis. The DACH cohort is the only
*commercial* signal feeding the ADR today. This brief expands the
commercial signal **internationally** — covering the most active
players, latest applied / industry studies (2022–2026), and potential
customers in three verticals (manufacturing & assembly, logistics &
warehousing, medical / construction / mining / defence) — so that the
ADR-007 review at Month 3 can cross-check whether the DACH-derived
6-axis shortlist (control rate, action space, obs modality, comm,
partner familiarity, safety) is internationally robust or DACH-local.

## Scope and method

- **Scope.** Industry deployments and applied / engineering studies
  from 2022–2026 outside the DACH cohort. Excludes pure simulation
  papers unless they ground a deployed system.
- **Method.** Web search across vendor press releases, trade-press
  reporting (Robotics 24/7, Robotics & Automation News, Robot Report,
  Wired, MIT Tech Review), peer-reviewed venues (RSS, ICRA, IROS,
  CoRL 2024–2026 where surfaced), and arXiv preprints. Not exhaustive;
  signal intended to widen the evidence basis, not replace it.
- **Non-manuscript content.** This brief contains industry / market
  signal that the project's read-only / manuscript-only rule does not
  apply to (the rule binds Tier-1 / Tier-2 *paper notes*, not ADR
  evidence). Citations are inline.

## 1. Active players by vertical (international, non-DACH)

### 1.1 Manufacturing & assembly

The international wave is converging on **humanoid-plus-fixed-arm
heterogeneous cells** sitting alongside mobile bases, with the
humanoid as the variable-control-rate, variable-modality
black-box-from-the-cell's-perspective partner. As of 2025:

- **BMW × Figure AI.** 15–30 Figure 02 humanoids inserting sheet-metal
  parts into welding fixtures at Spartanburg, US. Same cell as
  KUKA / ABB welding arms — explicit heterogeneous mix of humanoid
  + industrial arm + AGV.
- **Mercedes-Benz × Apptronik.** 10–20 Apollo humanoids piloted for
  tote delivery and material handling at European plants. Co-located
  with Mercedes' existing fixed-arm welding cells.
- **Tesla Optimus.** First Optimus deployment roadmap revealed at the
  2025 shareholder meeting; Fremont pilot-line insertion 2025, wider
  factory adoption targeted 2026.
- **Foxconn × Robust.AI / × NVIDIA / × Tesla.** Carter (Robust.AI)
  warehouse-automation robots being scaled at Foxconn; Foxconn Texas
  may deploy humanoids at the Houston AI plant.
- **Hyundai (Boston Dynamics).** Atlas-redesign begins 2026
  production; Hyundai pledged ₩125 trillion (KR) + $26 B (US through
  2028) for AI / robotics; humanoid factory deployment targeted 2028.
- **Toyota × Boston Dynamics × TRI.** Large Behavior Model on Atlas
  doing combined manipulation + locomotion sequences (announced 2025);
  Agility Digit signed RaaS with Toyota Manufacturing Canada Feb 2026
  (7+ commercial units active for RAV4 logistics).
- **BYD × UBTECH (China).** ~100–200 humanoids cited as the world's
  largest commercial humanoid deployment.
- **GXO × Agility.** 100+ Digit units contracted through 2026.

Implication: every confirmed deployment combines a **humanoid (slow,
chunked-action VLA / behavior-model partner) + industrial arm (fast,
torque-level) + mobile base (mid-rate)** in the same cell. This is
*exactly* the control-rate × action-space × obs-modality stacking
that ADR-007's literature shortlist (HetGPPO Tier-1 #23 + HARL Tier-1
#25) calls for, and it is internationally repeated, not DACH-local.

### 1.2 Logistics & warehousing

- **Amazon Robotics.** Crossed 1 M robots deployed mid-2025; Sparrow
  (multi-jointed arm + suction + CV) handles ~65 % of catalogue;
  Hercules / Titan drives, Robin / Cardinal sortation arms, Proteus
  autonomous mover, Sequoia pod system, Stowing — *the canonical
  international heterogeneous fleet*. Amazon's fleet-management layer
  explicitly orchestrates "mixed robot types operating in shared
  spaces" with dynamic path planning, task allocation, collision
  avoidance, throughput optimisation.
- **Symbotic.** 46 systems deployed, 37 completed sites, FY25 revenue
  $2.3 B, $22.5 B backlog. Integrated full-facility automation rather
  than mix-and-match heterogeneity, but the SymBots themselves operate
  across racking + sortation + pallet build with heterogeneous
  end-effectors.
- **Mujin, Dexterity, Covariant, Plus One Robotics.** Specialist
  AI-grasping vendors whose stations sit *inside* heterogeneous
  fleets owned by the integrator (e.g., DHL, FedEx, JD).
- **Boston Dynamics Stretch.** ~1000 cases / hour; Stretch + Spot +
  Atlas explicitly framed by BD as an "in-tandem with AGVs"
  heterogeneous warehouse fleet. DHL signed MoU for an additional
  1000-robot deployment in May 2025.
- **Robust.AI Carter at DHL Las Vegas.** 60 % productivity gain in
  weeks; foundation for DHL's 5-year strategic alliance (Mexico).
- **JD Logistics (China).** 5-year plan: **3 M robots, 1 M autonomous
  vehicles, 100 K drones** procured (announced Oct 2025).
- **Cainiao, Meituan, SF, Baidu Apollo, Haomo.** Driverless-delivery
  pilots in 100+ Chinese cities; mixed-vendor procurement (Cainiao
  procures from Neolix, Zelos in addition to in-house).
- **Agility Digit at Amazon.** Public deployment in fulfilment
  alongside Sparrow + Hercules — the heterogeneity is now humanoid
  + arm-station + drive-unit + AGV, all in the same building.

### 1.3 Medical, construction, mining, defence

- **Surgical (medical).** Da Vinci (Intuitive) vs Versius (CMR)
  vs Hugo (Medtronic) — modular **3–5 independent cart-mounted arms**
  with 360° wrists. Versius >15 K procedures by Feb 2024; explicit
  multi-arm collision-avoidance reports (more arm-collision events
  than da Vinci, per surgeon reports). Heterogeneity is per-arm
  module, per-vendor compliance, per-instrument actuation rate.
- **Construction.** Built Robotics, Boston Dynamics Spot + Stretch,
  Toggle (rebar), Canvas (drywall) — heterogeneous contact-rich
  manipulation in unstructured outdoor cells.
- **Mining.** Caterpillar Cat Command: 690 autonomous trucks at
  end-2024 → target 2 000+ by 2030; **explicit "mixed fleet" product**
  — Caterpillar's autonomy stack runs on Komatsu 930E trucks,
  cross-vendor — won contract for autonomy on 90+ trucks of
  competitive brand. Komatsu AHS dominant in large-scale projects.
  Vale confirmed fleet expansion deal Dec 2025. Global mining-robotics
  market $1.44 B (2024) → $3.70 B (2034).
- **Defence.** Anduril Lattice OS (multi-system battlefield network,
  $30.5 B valuation, $4 B raise pursued 2026 at $60 B), Shield AI
  Hivemind (autonomy on collaborative combat aircraft, $2 B / $12.7 B
  valuation), Helsing Europa (autonomous fighter), HX-2 drone swarms.
  Helsing Paris team explicitly building one-human-to-many-drones
  oversight; Anduril building one-to-10+ marshalling. *Heterogeneous
  multi-platform autonomy is the production stack*, not a research
  curiosity.

## 2. Per-axis applied evidence (2022–2026)

This is the section that directly tests whether ADR-007's literature
+ DACH shortlist holds up internationally. For each candidate axis,
I report whether the international applied / industry signal
**confirms**, **adds nuance to**, or **questions** inclusion.

### 2.1 Control-rate heterogeneity — **CONFIRMED, expanded**

The strongest international signal comes from VLA / foundation-model
literature *post-OpenVLA*:

- OpenVLA trained for control at 5–10 Hz; "empirically struggles
  with high-frequency data" (`openvla/openvla` README).
- **Real-Time Chunking (RTC)** [arXiv 2509.23224] and **Asynchronous
  Action Chunk Correction (A2C2)** [arXiv 2512.20188] explicitly
  reformulate chunk-switching as inpainting / tiny correction-head
  problems precisely because the slow-VLA + fast-controller mismatch
  is now treated as a load-bearing engineering problem.
- **FAVLA (Force-Adaptive Fast–Slow VLA)** [arXiv 2602.23648] and
  **Asynchronous Fast-Slow VLA** are both explicitly two-rate
  architectures.
- **VLSA / AEGIS** [arXiv 2512.11891] — plug-and-play CBF safety
  layer designed precisely to stitch onto a slow VLA partner.
- **SilentDrift** [arXiv 2601.14323]: a 1 mm per-step perturbation
  compounds to 5 cm over a 50-step chunk — confirms that the
  control-rate × chunk-size interaction is empirically the dominant
  failure mode, not just a theoretical concern.

This corroborates the v0.2 §3.4 #verify tag and HetGPPO §3.4 / HARL
§3.4 observation that control-rate is *absent from the HMARL
literature but structurally present in deployed stacks*. Industrial
deployments (BMW × Figure 02, Mercedes × Apptronik, Toyota × Atlas)
all couple a slow chunked-VLA / LBM partner with fast industrial
arms — the exact failure mode the project's manipulator (~500 Hz) +
mobile base (~50 Hz) + VLA partner (5–15 Hz) stack will encounter.
**The literature trend tightens, not weakens, the ≥20pp prior on this
axis** and explicitly extends it from arm/base-mismatch to
arm/VLA-partner mismatch.

### 2.2 Action-space heterogeneity — **CONFIRMED**

- **Open X-Embodiment / RT-X** (Oct 2023, expanded 2024): 22
  embodiments, 1 M+ trajectories, **embodiment scaling laws** showing
  *expanding embodiments yields more generalisation than expanding
  trajectory count for a fixed embodiment*. This is the strongest
  empirical claim on record that action-space heterogeneity is a
  first-class axis, not a nuisance variable.
- **XMoP** (cross-embodiment whole-body motion policy) — 70 % zero-shot
  on 7 commercial arms with no per-robot retraining.
- **CrossFormer** (already in the project's Tier-1 reading): the
  "no-positive-transfer-from-co-training" finding is replicated by
  multiple downstream evaluations.
- Industrial repetition: humanoid (28-DOF whole-body) + arm (7-DOF) +
  AGV (2-DOF base) deployments at BMW / Mercedes / Tesla /
  Foxconn / Amazon all manifest action-space heterogeneity at the
  cell level.

### 2.3 Observation-modality heterogeneity — **CONFIRMED, with industrial nuance**

- Visual–tactile fusion + autoencoder for peg-in-hole assembly
  (MDPI 2025) shows industrial peg-in-hole *requires* multi-modal
  heterogeneity (vision-only fails on contact state; force-only fails
  on macro pose).
- HIT (Harbin Institute of Technology) 2025 patent: visual + tactile +
  motion + torque NN ensembles, tactile-as-indirect-multi-DOF-force
  inferencer.
- *Multimodal fusion and VLMs survey for robot vision* (Elsevier,
  2025; arXiv 2504.02477) and *VLM/VLA for manipulation systematic
  review* (Elsevier, 2025) both treat modality heterogeneity as an
  axis with a non-trivial fusion-design space.
- 5G HRC review (OAE, 2025) specifically extends modalities to include
  haptic & proximity for human-robot collaboration over wireless.

### 2.4 Communication degradation (latency / drop / jitter) — **CONFIRMED with sharper numbers**

This is the axis where international signal has *moved* the prior
substantially since the v0.2 plan was written:

- **3GPP Release 16** standardised the 5G-as-virtual-TSN-bridge
  architecture (DS-TT / NW-TT); **Release 17** adds uplink time-sync
  and exposed northbound QoS. URLLC: 1 ms latency, 99.9999 %
  reliability when budgets allow.
- *Comparative Performance Evaluation of 5G-TSN Applications in
  Indoor Factory Environments* [arXiv 2501.12792]: explicit jitter
  measurements in factory deployments.
- Multi-robot synchronisation papers explicitly call out: "if two
  mobile robots are collaborating, jitter in the transmission can
  lead to varying offsets in their movements; depending on required
  synchronicity, this requires an upper bound on tolerable jitter."
- Ericsson, Qualcomm, IEEE VDE — three large-scale industrial trial
  reports between 2022 and 2025 all report that **commercial 5G-TSN
  for industrial IoT remains supply-limited** despite functioning
  prototypes — i.e., the axis is empirically present in *every*
  multi-vendor cell because deployments fall back to wired or
  best-effort wireless.
- AoI / AoCI literature (arXiv 2507.08429 etc.) is now treated as
  the canonical metric for "freshness of partner state" — directly
  mappable to v0.2's AoI proxy and ADR-008's HRS bundle.

### 2.5 Partner familiarity / black-box partner — **CONFIRMED, expanded**

- **MALMM (IROS 2025)** — multi-agent LLM with Supervisor /
  Planner / Coder / Executor agents for zero-shot manipulation.
- **LLM-Coordination (NAACL 2025 findings)** — LLM-vs-LLM zero-shot
  coordination evaluations, extends the Liu 2024 RSS line.
- **Leveraging LLM for Heterogeneous Ad Hoc Teamwork Collaboration**
  [arXiv 2406.12224] — explicit heterogeneous AHT with LLM.
- **TMLR Sept 2025** (Ruhdorfer et al.) on generalisation to novel
  partners under AHT / ZSC paradigms.
- **EMOS** (Embodiment-aware Heterogeneous Multi-robot OS) — LLM
  agents read URDFs + describe their own physical capabilities,
  enabling collaboration across embodiments.
- **CrossFormer's "no-positive-transfer"** finding is the strongest
  argument that *FM partners cannot be assumed cooperative* — the
  black-box framing is *not* an artificial constraint but reflects
  how these models behave in the wild.

### 2.6 Safety heterogeneity (per-vendor force / ISO compliance) — **CONFIRMED, regulatory shift since the plan**

This is where the strongest "since the plan was drafted" change is:

- **ISO/TS 15066 has been integrated into ISO 10218-2:2025**
  (replacing ISO 10218-1/2:2011). The technical specification is now
  *part of the binding standard*, not a supplement. ISO 10218-1:2025
  and ISO 10218-2:2025 add clearer functional-safety requirements,
  new classifications, and test methods.
- This regulatory consolidation makes per-vendor compliance variance
  more, not less, salient: a mixed-vendor cell now needs to satisfy
  the new 10218-2:2025 across heterogeneous controllers with
  different certification dates and different bake-ins of the old
  15066 force-pressure tables.
- Surgical (medical) signal: Versius + Hugo + da Vinci share the same
  OR but *differ in arm-collision frequency* (Versius / Hugo report
  more collisions due to independent BSU positioning). This is
  applied evidence that heterogeneous safety architectures across
  vendors produce different hazard rates **in deployment**, not just
  in spec.
- Mining signal: Caterpillar Cat Command running on Komatsu 930E
  trucks is the canonical commercial example of *one safety stack
  spanning multiple vendor controllers* — and the contract terms make
  this a paid, validated capability.
- Conformal-CBF + multi-robot literature has accelerated:
    - *Safe Decentralized Multi-Agent Control using Black-Box
      Predictors, Conformal Decision Policies, and CBFs*
      (Huriot & Sibai, arXiv 2409.18862 — *the project's Tier-1 #42*).
    - *Safe Probabilistic Planning for HRI using Conformal Risk
      Control* [arXiv 2603.10392].
    - *Formation-Aware Adaptive Conformalised Perception for Safe
      Leader–Follower MRS* [arXiv 2603.08958].
    - *CPED-NCBFs* [arXiv 2507.15022].
    - *Computationally and Sample Efficient Safe RL using ACP*
      [arXiv 2503.17678].
    - *Safe Task Planning for Language-Instructed MRS using CP*
      [arXiv 2402.15368].

This corroborates the DACH playbook §8.1 claim that **safety should
be a 6th candidate axis** — with international + regulatory
amplification beyond the DACH commercial signal alone.

## 3. Customer signal — who is buying

Concrete dollar / unit signal for "real production pain" that ADR-007
can cite alongside DACH integrator interviews:

- **Tier-1 automotive.** BMW (Spartanburg), Mercedes-Benz (EU), Tesla
  (Fremont), Hyundai (US 2028), Toyota (Canada), BYD (CN). All with
  named units already in cells.
- **3PL & e-commerce logistics.** Amazon (>1 M robots), DHL (1 K
  Boston Dynamics MoU + 5-year Robust.AI alliance in Mexico), GXO
  (100+ Digit), JD Logistics (3 M robots / 1 M AVs / 100 K drones),
  Cainiao + Meituan + SF + Baidu Apollo + Haomo (multi-vendor
  procurement).
- **Contract manufacturing.** Foxconn (Robust.AI partner, Texas
  pilot, COMPUTEX 2025 robotics announcements), Jabil (humanoid
  scale-up commentary).
- **Surgical.** Intuitive (da Vinci, market leader), CMR Versius
  (>15 K procedures Feb 2024), Medtronic Hugo, Asensus.
- **Mining.** Vale (CAT fleet expansion Dec 2025), Komatsu AHS
  customers, Caterpillar 690 trucks → 2 000 by 2030.
- **Defence.** US DoD via Replicator, Anduril, Shield AI; European
  defence via Helsing.
- **Investment scale.** Robotics deal value $7.3 B in H1 2025;
  mining-robotics market $1.44 B (2024) → $3.70 B (2034); Symbotic
  $22.5 B backlog FY25.

The DACH playbook § 9 go/no-go rule (≥6/8 conversations name
coordination pain + at least one past spend) is highly unlikely to
fail given that *every* international vertical above is paying
multi-billion dollars for exactly the heterogeneity-coordination
problem the plan describes.

## 4. ADR-007 implications

This brief does **not** modify ADR-007 (per request). The implications
flagged for the Month-3 review:

1. **Option B (5-axis literature shortlist) → Option B′ (6-axis
   shortlist with safety) is internationally supported.** The DACH
   playbook §8.1 6-axis list (control rate / action space / obs
   modality / comm / partner familiarity / safety) matches the
   international applied signal. The empirical extension to add
   *safety as a standalone §3.4 axis* is no longer DACH-local.

2. **Control-rate axis is empirically over-determined for inclusion.**
   Independent of the Phase-0 spike result, the international applied
   literature (RTC, A2C2, FAVLA, Asynchronous Fast-Slow VLA, VLSA,
   SilentDrift) and the deployed humanoid+arm cells (BMW, Mercedes,
   Toyota, Foxconn) treat slow-VLA / fast-controller mismatch as the
   dominant failure mode. The ≥20pp gap test will likely pass on the
   first try; the spike may need to be designed for *axis isolation*
   rather than gap demonstration (control-rate vs action-space
   confound, per ADR-007 risks).

3. **Communication-degradation axis has tighter numerical priors than
   the v0.2 plan presumed.** 3GPP Release 17, URLLC 1 ms / 99.9999 %
   reliability, and explicit factory-jitter ablations (arXiv
   2501.12792) give per-task numeric bounds (jitter: μs–ms;
   drop: 10⁻⁶; latency: 1–10 ms wired-equivalent). ADR-006's
   "explicit numeric bounds per benchmark task" gap can be filled
   from public 5G-TSN industrial trial data.

4. **Safety axis benefits from a regulatory-window rationale.**
   ISO 10218-2:2025 absorbing ISO/TS 15066 is *recent*. Mixed-vendor
   cells are now compliance-heterogeneous in a way they were not when
   the v0.2 plan was drafted. This is a non-trivial argument for
   safety-as-axis even if the Phase-0 spike's ≥20pp gap is borderline.

5. **A potential 7th candidate axis surfaces (do *not* add without
   validation): control-loop hierarchy / chunk-async coupling.**
   Reading RTC + A2C2 + Asynchronous Fast-Slow VLA as a cluster, the
   axis is not exactly "control rate" — it is "two policies operating
   at decoupled clocks with chunk inpainting between them." This may
   subsume control-rate as a special case. **Do not pre-register as
   a 7th spike axis** — flag it as an open question for the post-
   spike axis-list review.

6. **Customer-signal threshold is over-cleared internationally.** If
   the DACH §9 rule were applied to the international signal, it
   would PROCEED-WITHOUT-ADJUSTMENTS by a wide margin. This does not
   replace the DACH synthesis (which is the contractually-committed
   gate), but it provides a robust prior that the DACH outcome will
   not collapse the project.

## 5. Open questions for ADR-007 review at Month 3

- Does the Phase-0 control-rate spike isolate frequency from
  action-space dimensionality (per ADR-007 risk #2)? International
  literature suggests deliberately running a single-embodiment, two-
  clock test before claiming control-rate as standalone.
- Should observation-modality be split into (a) sensor-suite
  heterogeneity and (b) representation-format heterogeneity (raw
  pixels vs proprio scalars)? The visual-tactile fusion literature
  treats these as different problems; ADR-007 currently treats them
  as one axis.
- Should "control-loop hierarchy / chunk-async coupling" be added as
  a 7th axis after the spikes, or absorbed into control-rate?
- Does the safety axis decompose into (a) per-vendor force-limit
  compliance (ISO/TS 15066-style) and (b) per-vendor functional-
  safety SIL/PL ratings (ISO 10218-2:2025-style)? They have
  different hazard-rate profiles.
- Is the international customer signal sufficient to *unblock*
  ADR-007 ahead of the DACH synthesis if the synthesis slips? (Per
  the locking rule, ADR-007 cites Tier-1 #23 + #25 + DACH playbook;
  this brief adds international corroboration but does not replace
  the playbook §9 decision-rule application.)

## Sources

### Manufacturing & assembly
- [Industrial Robots Research Report 2025 — BMW, Mercedes, Tesla pilot factory deployments (BusinessWire, Nov 2025)](https://www.businesswire.com/news/home/20251107524093/en/Industrial-Robots-Research-Report-2025-Moving-from-Automation-to-Autonomy---Humanoid-Collaborative-AI-Driven-Robotics-Reshape-Manufacturing-as-BMW-Mercedes-Benz-Tesla-Pilot-Factory-Deployments---ResearchAndMarkets.com)
- [Hyundai Motor Group AI Robotics Strategy (Hyundai News)](https://www.hyundainews.com/releases/4664)
- [Foxconn / Robust.AI Carter manufacturing partnership (May 2025)](https://www.foxconn.com/en-us/press-center/press-releases/latest-news/1597)
- [Apptronik Apollo at Mercedes-Benz](https://humanoid.press/database/apptronik-apollo-factory-collaboration-robot/)
- [Figure vs Apptronik vs Agility Robotics (Sacra)](https://sacra.com/research/figure-vs-apptronik-vs-agility-robotics/)
- [Boston Dynamics × Toyota Research Institute LBM on Atlas](https://pressroom.toyota.com/ai-powered-robot-by-boston-dynamics-and-toyota-research-institute-takes-a-key-step-towards-general-purpose-humanoids/)
- [Humanoid Robots in Industrial Manufacturing — what they can / can't do in 2026](https://www.evsint.com/humanoid-robots-industrial-manufacturing-2026/)
- [Jabil — Humanoid Robots: Mass Adoption hinges on Affordability and Scale](https://jabil.com/blog/humanoid-robots-mass-adoption.html)

### Logistics & warehousing
- [Amazon — Sparrow handles millions of diverse products (Amazon News)](https://www.aboutamazon.com/news/operations/amazon-introduces-sparrow-a-state-of-the-art-robot-that-handles-millions-of-diverse-products)
- [Amazon Robotics fleet (Amazon News)](https://www.aboutamazon.com/news/operations/amazon-robotics-robots-fulfillment-center)
- [Amazon's 1 M robots milestone (Metaintro)](https://www.metaintro.com/blog/amazon-1-million-robots-600000-warehouse-jobs-workers-2026)
- [Symbotic — warehouse-automation system overview](https://www.symbotic.com/symbotic-system/)
- [Symbotic FY25 results & competitive landscape (Motley Fool)](https://www.fool.com/investing/2025/06/13/is-symbotic-stock-a-buy-as-ai-transforms-warehouse/)
- [Boston Dynamics warehouse robotics podcast / Stretch / Spot / Atlas](https://thenewwarehouse.com/2025/09/10/624-bringing-robotics-to-the-warehouse-floor-with-boston-dynamics/)
- [DHL × Boston Dynamics MoU 1000 robots (May 2025)](https://group.dhl.com/en/media-relations/press-releases/2025/dhl-group-signs-mou-with-boston-dynamics-and-accelerates-cross-business-automation-strategy.html)
- [DHL × Robust.AI 5-year alliance (Mexico, Dec 2025)](https://www.businesswire.com/news/home/20251202202650/en/DHL-Supply-Chain-Announces-Five-year-Strategic-Alliance-with-Robust.AI-to-Drive-the-Next-Generation-of-Logistics-Automation-in-Mexico)
- [JD Logistics 5-year millions-of-robots plan (TechNode, Oct 2025)](https://technode.com/2025/10/27/jd-logistics-unveils-five-year-plan-to-deploy-millions-of-robots-autonomous-vehicles-and-drones/)
- [Warehouse robots 2026 — Amazon, Ocado, Symbotic](https://unteachablecourses.com/warehouse-robots-2026/)
- [DHL — Robot Integration Platform (one layer for many robots)](https://www.dhl.com/global-en/delivered/innovation/so-many-robots-one-simple-integration-layer.html)

### Medical / construction / mining / defence
- [Versius vs da Vinci comparative analysis (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10959786/)
- [SAGES review of upcoming multi-visceral robotic surgery systems (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11615118/)
- [Multimodular Hugo RAS / Versius CMR for pelvic surgery (ResearchGate)](https://www.researchgate.net/publication/374293281_Multimodular_robotic_systems_Hugo_RAS_and_Versius_CMR_for_pelvic_surgery_tasks_and_perspectives_from_the_bed-side_assistant)
- [Caterpillar 2 000 autonomous mining trucks by 2030 (International Mining)](https://im-mining.com/2025/11/07/caterpillar-sets-out-to-hit-over-2000-autonomous-mining-trucks-by-2030/)
- [Caterpillar mixed-fleet autonomy on Komatsu 930E (OEM Off-Highway)](https://www.oemoffhighway.com/electronics/smart-systems/automated-systems/news/20977217/caterpillar-inc-torc-robotics-helps-caterpillar-develop-autonomous-mining-solution)
- [Vale × Caterpillar fleet expansion (Dec 2025)](https://im-mining.com/2025/12/08/vale-confirms-autonomous-truck-fleet-expansion-deal-with-caterpillar/)
- [Mining Robotics Market $3.70 B by 2034 (Precedence Research)](https://www.precedenceresearch.com/mining-robotics-market)
- [Anduril (corporate)](https://www.anduril.com/)
- [Shield AI $2 B raise / Hivemind (TheNextWeb)](https://thenextweb.com/news/shield-ai-2-billion-hivemind-autonomous-defence)
- [The future of autonomous warfare in Europe (MIT Tech Review, Jan 2026)](https://www.technologyreview.com/2026/01/06/1129737/autonomous-warfare-europe-drones-defense-automated-kill-chains/)
- [Anduril × Shield AI YFQ-44A flight (Defence Industry Europe)](https://defence-industry.eu/andurils-yfq-44a-completes-flight-with-dual-mission-autonomy-software-from-anduril-and-shield-ai-in-cca-test/)

### Per-axis applied evidence
- [Open X-Embodiment / RT-X arXiv 2310.08864](https://arxiv.org/abs/2310.08864)
- [Open X-Embodiment GitHub](https://github.com/google-deepmind/open_x_embodiment)
- [OpenVLA — control frequency ~5–10 Hz, action chunking limitations](https://github.com/openvla/openvla)
- [Real-Time Chunking for VLA action chunks — arXiv 2509.23224](https://arxiv.org/html/2509.23224v1)
- [Asynchronous Fast-Slow VLA — arXiv 2512.20188](https://arxiv.org/html/2512.20188v1)
- [FAVLA — Force-Adaptive Fast–Slow VLA, arXiv 2602.23648](https://arxiv.org/html/2602.23648)
- [VLSA — plug-and-play CBF safety constraint layer, arXiv 2512.11891](https://arxiv.org/html/2512.11891v1)
- [SilentDrift — chunk-perturbation backdoor, arXiv 2601.14323](https://arxiv.org/html/2601.14323)
- [Multimodal fusion + VLM survey for robot vision, arXiv 2504.02477](https://arxiv.org/html/2504.02477v1)
- [Visual–tactile fusion + SAC peg-in-hole assembly (MDPI Machines)](https://www.mdpi.com/2075-1702/13/7/605)
- [Multi-modal AI for robot assembly — patent landscape (PatSnap)](https://www.patsnap.com/resources/blog/articles/multi-modal-ai-for-robot-assembly-50-patents-analyzed/)
- [5G-TSN integration architecture (Ericsson)](https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/5g-tsn-integration-for-industrial-automation)
- [Comparative 5G-TSN performance in indoor factories — arXiv 2501.12792](https://arxiv.org/html/2501.12792v2)
- [Bridging the gap: 5G-TSN integration for industrial robotic communication (IEEE/VDE)](https://ieeexplore.ieee.org/document/10477097)
- [Qualcomm — Ultra-Reliable Low-Latency 5G for Industrial Automation](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/ultra-reliable-low-latency-5g-for-industrial-automation.pdf)
- [Age of Information optimisation in laser-charged UAV-IoT, arXiv 2507.08429](https://arxiv.org/html/2507.08429v1)
- [LLM-Coordination NAACL 2025 findings](https://aclanthology.org/2025.findings-naacl.448.pdf)
- [MALMM — Multi-Agent LLMs for Zero-Shot Manipulation, IROS 2025](https://malmm1.github.io/assets/IROS_2025_malmm_v8.pdf)
- [LLM for Heterogeneous Ad Hoc Teamwork — arXiv 2406.12224](https://arxiv.org/html/2406.12224v1)
- [TMLR Sept 2025 — generalisation to novel partners under AHT/ZSC (Ruhdorfer et al.)](https://www.collaborative-ai.org/publications/ruhdorfer25_tmlr.pdf)

### Safety / conformal
- [ISO 10218-1:2025 / 10218-2:2025 update absorbing ISO/TS 15066 (Standard Bots)](https://standardbots.com/blog/collaborative-robot-safety-standards)
- [ISO 10218 & ISO/TS 15066 explained for integrators (AMD Machines)](https://amdmachines.com/blog/robot-safety-standards-iso-10218-and-ts-15066-explained/)
- [ISO/TS 15066:2016 (ISO catalogue)](https://www.iso.org/standard/62996.html)
- [Huriot & Sibai — Safe Decentralized MAS using Black-Box Predictors + Conformal Decision + CBF, arXiv 2409.18862](https://arxiv.org/html/2409.18862)
- [Safe Probabilistic Planning for HRI using Conformal Risk Control, arXiv 2603.10392](https://arxiv.org/html/2603.10392v1)
- [Formation-Aware Adaptive Conformalised Perception for Safe Leader–Follower MRS, arXiv 2603.08958](https://arxiv.org/html/2603.08958)
- [CPED-NCBFs — Conformal Prediction for Expert-Demo Neural CBFs, arXiv 2507.15022](https://arxiv.org/html/2507.15022)
- [Computationally and Sample Efficient Safe RL using ACP, arXiv 2503.17678](https://arxiv.org/html/2503.17678)
- [Safe Task Planning for LLM-instructed MRS using CP, arXiv 2402.15368](https://arxiv.org/html/2402.15368v1)

### Cross-cutting industrial / market
- [Industrial robotics 2025 trends, figures, and global outlook (Robotnik)](https://robotnik.eu/industrial-robotics-in-2025-trends-figures-and-global-outlook/)
- [2025 robotics trends — humanoids enter commercial use (Robotics 24/7)](https://www.robotics247.com/article/2025-robotics-trends-humanoids-enter-commercial-use-logistics-and-automation-rise)
- [Omdia Market Radar — General-purpose Embodied Intelligent Robots, 2026](https://omdia.tech.informa.com/om143809/omdia-market-radar-generalpurpose-embodied-intelligent-robots-2026)
- [Multi-robot collaborative manipulation framework (Frontiers Robotics & AI 2025)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1585544/full)
- [Cooperative planning for physically interacting heterogeneous robots (Frontiers 2024)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1172105/full)
- [Toward Universal Embodied Planning in Scalable Heterogeneous Field Robots (J. Field Robotics 2025)](https://onlinelibrary.wiley.com/doi/10.1002/rob.22522)
- [Multi-Robot Systems Survey — Architectures, Performances (TechRxiv 2025)](https://www.techrxiv.org/users/968140/articles/1339550/master/file/data/Multi_Robot_Systems_Survey_TechRxiv_2025/Multi_Robot_Systems_Survey_TechRxiv_2025.pdf?inline=true)
- [Intent-driven LLM ensemble planning for flexible multi-robot manipulation (Frontiers 2026)](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2026.1727433/full)
- [ZTE 5G-A EasyOn·Robot private network at WAIC 2025](https://www.zte.com.cn/global/about/news/zte-5g-a-easyon-robot-private-network-facilitates-multi-robot-collaboration-at-waic-2025.html)
