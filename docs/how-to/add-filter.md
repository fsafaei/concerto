# How-to: Add a new safety filter

!!! note "Phase-0 placeholder"
    Full recipe added once `concerto.safety` is implemented in M3.

## Steps (sketch)

1. Implement the `SafetyFilter` protocol from `concerto.api.safety`.
2. Register the filter in `concerto.safety.__init__`.
3. Write a property test verifying the feasibility / objective-bound
   invariant (see `plan/08-quality-and-process.md §3`).

*(Full content added in M3.)*
