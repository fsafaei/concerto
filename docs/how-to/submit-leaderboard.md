# How-to: Submit a leaderboard entry

The leaderboard accepts entries produced under the CHAMBER-Bench
v1.0 protocol (ADR-027): preregistered runs, verifiable result
bundles (ADR-028), the public partner set, and per-task reporting.
The end-to-end flow is: **fork → preregister → run the preregistered
cells against the public partner set → open a PR containing the
bundle → CI runs `chamber-eval verify` → maintainer spot-check →
merge.**

Before starting, read the
[evaluation protocol](../explanation/evaluation-protocol.md) — it
defines the terms used below (admission, preregistration, result
bundle, checkpoint-selection rule) — and reproduce at least one
existing row via
[Reproduce the results](reproduce-results.md) so you know your
environment matches.

## 1. Fork and set up

```bash
git clone https://github.com/<you>/concerto.git
cd concerto
git fetch --tags        # the preregistration gate resolves tags
uv sync --group dev
```

## 2. Register your method

Your method controls the ego arm. Implement it as a registered
policy (for learned methods, a checkpoint-loading policy; for
scripted methods, a policy class) so `chamber-eval run --policy
<your-method-id>` can construct it. See
[Add a partner](add-partner.md) for the registry pattern — ego
policies follow the same shape. Your PR must include the method
code (or a pinned dependency on it) so the maintainer and CI can
re-run your cells.

## 3. Preregister your cells

Copy the campaign preregistration for the task you target
(`spikes/preregistration/benchmark/cocarry_baselines_v1.yaml` or
`handover_baselines_v1.yaml`) and add your method as a row: method
id, the same seeds and episode counts as the existing rows, the same
estimator, and — for learned methods — the checkpoint-selection rule
(per seed, highest stress-compliant success on the held-out
validation partner; never on the evaluation set). Commit the YAML
and create a signed tag:

```bash
git tag -s prereg-<task>-<your-method>-<date>
```

Editing a preregistration after its tag exists is refused by the
tooling — re-issue under a new tag instead.

## 4. Run the preregistered cells against the public partner set

One `chamber-eval run` per cell, mirroring the committed rows (see
each bundle's `REPRO.txt` for the exact shape):

```bash
uv run chamber-eval run --task cocarry --policy <your-method-id> \
  --partner-set cocarry_partners@v1 --exclude-member imp_nominal \
  --seeds 5 --episodes 50 --out spikes/results/benchmark/cocarry-v1/<your-method>-<date> \
  --prereg spikes/preregistration/benchmark/<your-prereg>.yaml
uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/<your-method>-<date>
```

Requirements the tooling enforces:

- **Clean tree.** Bundles from a locally-modified checkout are
  stamped dirty and are leaderboard-ineligible.
- **Public partner set only.** You run against the public members;
  private members' parameters are withheld by design (their identity
  hashes are published, so the maintainer's spot-check is
  verifiable). Do not pass `--include-private`.
- **Held-out validation partner.** Keep the campaign's
  `--exclude-member` flags exactly — the excluded member is the
  checkpoint-selection partner and must not appear in evaluation.
- **All seeds.** Single-seed submissions are not accepted
  (ADR-027 reporting rules).

## 5. Open the PR

The PR contains: the tagged preregistration YAML, your method code,
the result bundle directory (or directories), and one line per
bundle added to
`spikes/results/benchmark/<task>/LEADERBOARD_BUNDLES.txt`. Then run
the renderer so the README table includes your row:

```bash
uv run python scripts/render_leaderboard_table.py
```

CI re-runs the full `chamber-eval verify` check table on every
listed bundle and fails on any drift between the bundles and the
rendered table. Large learned-method checkpoints do not go in git —
attach them as a release asset on your fork or a public download,
reference the URL and SHA-256 in the PR, and keep the
`local://artifacts/...` URIs in `REPRO.txt` accurate.

## 6. Maintainer verification and merge

Before merge the maintainer may spot-check your method against the
**private partner split** — the withheld 30% of the partner set —
to detect overfitting to the public members. A public/private gap is
reported alongside your row if found; a method that collapses on
private partners is not merged. Method class labels (baseline,
oracle reference, non-AHT upper anchor) are assigned per ADR-027's
honest-label rules, not self-declared.

Contributions are accepted under the project CLA (ADR-012) with DCO
sign-off and signed commits — see
[CONTRIBUTING](https://github.com/fsafaei/concerto/blob/main/CONTRIBUTING.md).
