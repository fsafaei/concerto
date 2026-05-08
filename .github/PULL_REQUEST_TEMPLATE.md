## Summary

<!-- One paragraph: what changed and why. -->

## ADR references

<!-- Cite the ADR(s) this PR implements or touches.
     Example: ADR-001 §Decision item 2; ADR-007 Stage 1 AS spike. -->

-

## Test plan

<!-- How was this validated? Reference new tests by path. -->

- [ ] Unit tests added/updated
- [ ] Property tests (Hypothesis) added/updated where applicable
- [ ] Integration tests added/updated
- [ ] `make verify` green locally
- [ ] `make verify-no-ai-mentions` green locally

## Risk

<!-- Known risks, edge cases, or rollback plan. Cross-reference any open
     risks in the milestone's plan/*.md §Risk register if applicable. -->

## Reproducibility note

<!-- If this PR produces a results artefact (a spike, a benchmark output,
     a published number), name the `scripts/repro/<artefact>.sh` that
     reproduces it from a clean checkout. -->

## Checklist

- [ ] Conventional Commits format on every commit message
- [ ] Every public symbol has a docstring with an ADR reference (where ADR-bearing)
- [ ] No `from chamber` imports inside `src/concerto/`
- [ ] No private (`_`) imports from `mani_skill`
- [ ] Documentation updated in the same PR (Diátaxis quadrant)
