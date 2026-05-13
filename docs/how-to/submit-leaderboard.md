# How-to: Submit a leaderboard entry

!!! note "Phase-0 placeholder"
    The populated leaderboard lands with M5 once the Stage-1 (AS + OM)
    spike rows are in. This page documents the submission protocol so
    that external contributors can prepare entries against the same
    contract used by the in-tree spikes.

Leaderboard entries follow the same preregistration discipline as every
other spike in the project: the hypothesis, threshold, and comparison
conditions are committed before the run starts, the YAML is tagged in
git, and the resulting rows are rendered into the leaderboard table by
`chamber-render-tables`.

## Protocol (sketch)

1. Copy the nearest existing pre-registration YAML from
   `spikes/preregistration/` as a template for your method's entry.
2. Edit the hypothesis, threshold, comparison conditions, and the
   `method:` name that will appear in the leaderboard row.
3. Commit the YAML and create a signed git tag of the form
   `leaderboard-<method>-<stage>-<date>`. **Editing the YAML after
   the tag exists is a project anti-pattern** — re-tag with a new YAML
   instead.
4. Run the spike via `chamber-spike run --axis <axis>` against the
   tagged YAML. See [Run a spike with a custom
   hypothesis](run-spike.md) for the end-to-end flow, including the
   M2 comm-degradation surface that the Stage-2 CM rows consume.
5. Open a PR that adds the tagged YAML and the result archive under
   `spikes/results/`. The CI gate re-renders the leaderboard table
   from the tagged result archives; no hand-edit of the README is
   required.

External contributors who do not have write access to the upstream
repository can attach the signed result archive to a PR as a release
asset and reference it from the preregistration YAML.

*(Full content — submission template, JSON schema for the result
archive, and the rliable-style aggregate reporting contract — added
in M5.)*
