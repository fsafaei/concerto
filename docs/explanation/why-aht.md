# Why heterogeneous ad-hoc teamwork?

Most multi-robot manipulation benchmarks assume identical embodiments and
shared training. Real-world deployment — hospital logistics, disaster response,
factory retooling — routinely pairs robots with incompatible action spaces,
differing sensor modalities, and no shared policy checkpoints.

CONCERTO and CHAMBER focus on the hardest variant of this problem: the
ego agent's partner is **opaque** (no policy access), **heterogeneous**
(different morphology and action frequency), and **ad-hoc** (no prior
joint training). The six axes that make this hard are: action-space heterogeneity (AS),
observation-modality heterogeneity (OM), control-rate mismatch (CR),
communication degradation (CM), partner familiarity (PF), and safety (SA).
See ADR-007 for the axis selection rationale and staging plan.

The four named precedents CONCERTO differentiates against — Liu 2024 RSS,
COHERENT, Huriot–Sibai 2025, and Singh 2024 — each solve a subset of
these axes. No prior system addresses all six simultaneously.
