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
`chamber-render-tables`. The pre-registration YAML schema, the result
archive schema, and the bootstrap / HRS pipeline live in
[`chamber.evaluation`](../reference/api.md):
`results.py` defines the Pydantic models for `SpikeRun` /
`EpisodeResult` / `LeaderboardEntry`; `prereg.py` validates the YAML
and verifies the git-tag SHA per
[ADR-007 §Discipline](https://github.com/fsafaei/concerto/blob/main/adr/ADR-007-heterogeneity-axis-selection.md);
`bootstrap.py` ships the cluster + paired-cluster bootstrap used for
the ≥20 pp gap test (reviewer P1-9); `hrs.py` computes the per-axis
HRS vector and the headline scalar per
[ADR-008 §Decision](https://github.com/fsafaei/concerto/blob/main/adr/ADR-008-hrs-bundle.md).

## Protocol

1. Copy the nearest existing pre-registration YAML from
   `spikes/preregistration/` as a template for your method's entry.
   The YAML schema is validated by
   `chamber.evaluation.prereg.load_prereg`; the required fields are
   `axis`, `condition_pair` (homogeneous / heterogeneous ids),
   `seeds`, `episodes_per_seed`, `estimator`, `bootstrap_method`
   (defaults to `cluster`), `failure_policy`, and `git_tag`.
2. Edit the hypothesis, threshold, comparison conditions, and the
   `method:` name that will appear in the leaderboard row. Set
   `bootstrap_method: cluster` unless you have a written reason to
   use `hierarchical` (alias) or `iid` (power-calc only — not
   admitted to the leaderboard).
3. Commit the YAML and create a signed git tag of the form
   `leaderboard-<method>-<stage>-<date>`. **Editing the YAML after
   the tag exists is a project anti-pattern** —
   `chamber.evaluation.prereg.verify_git_tag` refuses any submission
   whose on-disk blob SHA disagrees with the SHA stored at the tag,
   so re-tag with a new YAML instead.
4. Run the spike via `chamber-spike run --axis <axis>` against the
   tagged YAML. See [Run a spike with a custom
   hypothesis](run-spike.md) for the end-to-end flow, including the
   M2 comm-degradation surface that the Stage-2 CM rows consume.
5. Compose the leaderboard entry with `chamber-eval <spike_run.json>
   --method-id <id> --output entry.json`. The pipeline (cluster
   bootstrap → paired-cluster gap test → HRS vector → HRS scalar)
   uses `concerto.training.seeding.derive_substream` for deterministic
   resampling; identical inputs and seed produce byte-identical
   outputs.
6. Render the headline tables with `chamber-render-tables
   --leaderboard entry.json` and (if your spike emits the ADR-014
   three-table safety report) `chamber-render-tables --safety-report
   three_tables.json [--fmt latex]`.
7. Open a PR that adds the tagged YAML and the result archive under
   `spikes/results/`. The CI gate re-renders the leaderboard table
   from the tagged result archives; no hand-edit of the README is
   required.

External contributors who do not have write access to the upstream
repository can attach the signed result archive to a PR as a release
asset and reference it from the preregistration YAML.

The HRS vector is emitted alongside the scalar on every entry per
ADR-008 §Decision (reviewer P1-8); the renderer refuses entries that
carry only the scalar, so consumers can always recompute the headline
under a different weighting without re-running the spikes.
