# Verified-install record

The reviewer simulation for the v0.8.0 / CHAMBER-Bench v1.0 release
(executor protocol CB-08 §4): a fresh clone in a fresh Python
environment, the README install and five-minute quickstart run
verbatim, and one leaderboard bundle downloaded from the hosted
Hugging Face dataset and re-verified — on a checkout that never built
the repository before.

- **Date:** 2026-07-06
- **Source:** `https://github.com/fsafaei/concerto.git` at commit
  `5fc28f3db5e3c7824e6d2666278a6797a55a6ede` (the tree released as
  v0.8.0; the tag itself adds only the release version-bump commit,
  so the installed distribution below still reports 0.7.0 — CI re-runs
  the same quickstart commands on the tagged commit via the
  `readme-quickstart` and `smoke-eval` jobs)
- **Platform:** Linux 6.17.0-29-generic x86_64, Python 3.12.3
- **Outcome:** all steps passed — quickstart bundle `verify: PASS
  (20/20 checks)`, shipped row `verify: PASS (36/36 checks)`,
  hosted-download row `verify: PASS (36/36 checks)`

The transcript below is unedited except that the run's working
directory is normalized to `/home/reviewer/work`.

```console
# Environment preamble (not part of the README path): an empty
# directory and a fresh Python virtual environment providing only
# python + pip, so the README's own commands do the rest.
$ python3 --version
Python 3.12.3
$ python3 -m venv bootstrap-venv && source bootstrap-venv/bin/activate

# --- README §Install, verbatim ---
$ git clone https://github.com/fsafaei/concerto.git
Cloning into 'concerto'...
$ cd concerto
$ pip install uv && uv sync --group dev
 + webcolors==25.10.0
 + werkzeug==3.1.8
 + wrapt==2.1.2

# --- README §Five-minute quickstart, step 1, verbatim ---
$ uv run chamber-eval run --task mpe_cooperative_push --policy random \
    --partner scripted_heuristic --seeds 2 --episodes 5 --out out/quickstart-bundle
warning: `VIRTUAL_ENV=/home/reviewer/work/bootstrap-venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
chamber-eval run: wrote mpe_cooperative_push@v1 bundle to out/quickstart-bundle (10 episodes; success IQM 0.667 [0.000, 1.000])
$ uv run chamber-eval verify out/quickstart-bundle
warning: `VIRTUAL_ENV=/home/reviewer/work/bootstrap-venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
PASS  bundle:schema                          v3 ResultBundle
PASS  bundle:eligibility                     clean working tree
PASS  manifest:REPRO.txt                     sha256 match
PASS  manifest:episodes_seed0.jsonl          sha256 match
PASS  manifest:episodes_seed1.jsonl          sha256 match
PASS  manifest:partners.json                 sha256 match
PASS  manifest:membership                    no unmanifested files
PASS  sha256sums:REPRO.txt                   sha256 match
PASS  sha256sums:bundle.json                 sha256 match
PASS  sha256sums:episodes_seed0.jsonl        sha256 match
PASS  sha256sums:episodes_seed1.jsonl        sha256 match
PASS  sha256sums:partners.json               sha256 match
PASS  summary:n_episodes                     10 episodes (schedule expects 10)
PASS  summary:success_mean                   recomputed 0.600000 vs stated 0.600000
PASS  summary:success_iqm                    recomputed 0.666667 vs stated 0.666667
PASS  summary:success_ci_low                 recomputed 0.000000 vs stated 0.000000
PASS  summary:success_ci_high                recomputed 1.000000 vs stated 1.000000
PASS  partner:scripted_heuristic:registered  scripted_heuristic
PASS  partner:hashes                         identity hashes match
PASS  prereg                                 unpreregistered run (no tag referenced)
verify: PASS (20/20 checks)

# --- README §Five-minute quickstart, step 2, verbatim ---
$ uv run chamber-eval verify spikes/results/benchmark/cocarry-v1/ref-script-2026-07-05
warning: `VIRTUAL_ENV=/home/reviewer/work/bootstrap-venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
PASS  bundle:schema                                        v3 ResultBundle
PASS  bundle:eligibility                                   clean working tree
PASS  manifest:REPRO.txt                                   sha256 match
PASS  manifest:episodes_seed0.jsonl                        sha256 match
PASS  manifest:episodes_seed1.jsonl                        sha256 match
PASS  manifest:episodes_seed2.jsonl                        sha256 match
PASS  manifest:episodes_seed3.jsonl                        sha256 match
PASS  manifest:episodes_seed4.jsonl                        sha256 match
PASS  manifest:partners.json                               sha256 match
PASS  manifest:prereg.yaml                                 sha256 match
PASS  manifest:membership                                  no unmanifested files
PASS  sha256sums:REPRO.txt                                 sha256 match
PASS  sha256sums:bundle.json                               sha256 match
PASS  sha256sums:episodes_seed0.jsonl                      sha256 match
PASS  sha256sums:episodes_seed1.jsonl                      sha256 match
PASS  sha256sums:episodes_seed2.jsonl                      sha256 match
PASS  sha256sums:episodes_seed3.jsonl                      sha256 match
PASS  sha256sums:episodes_seed4.jsonl                      sha256 match
PASS  sha256sums:partners.json                             sha256 match
PASS  sha256sums:prereg.yaml                               sha256 match
PASS  summary:n_episodes                                   1750 episodes (schedule expects 1750)
PASS  summary:success_mean                                 recomputed 1.000000 vs stated 1.000000
PASS  summary:success_iqm                                  recomputed 1.000000 vs stated 1.000000
PASS  summary:success_ci_low                               recomputed 1.000000 vs stated 1.000000
PASS  summary:success_ci_high                              recomputed 1.000000 vs stated 1.000000
PASS  partner:ego:ref_script_cocarry_impedance:registered  cocarry_impedance
PASS  partner:imp_stiff_low:registered                     cocarry_impedance
PASS  partner:imp_stiff_high:registered                    cocarry_impedance
PASS  partner:imp_damp_low:registered                      cocarry_impedance
PASS  partner:imp_damp_high:registered                     cocarry_impedance
PASS  partner:imp_lag_bounded:registered                   cocarry_impedance
PASS  partner:imp_blend_b:registered                       cocarry_impedance
PASS  partner:imp_blend_c:registered                       cocarry_impedance
PASS  partner:hashes                                       identity hashes match
PASS  prereg:tag                                           tag prereg-cocarry-baselines-v1-rev2-2026-07-05 exists
PASS  prereg:blob                                          bundle copy matches verified blob SHA
verify: PASS (36/36 checks)

# --- Hosted-bundle verify: download one leaderboard bundle from the
# Hugging Face dataset (no repo build involved) and verify it ---
$ uvx --from 'huggingface_hub[cli]' hf download fsafaei/chamber-bench-leaderboard-bundles --repo-type dataset --include 'benchmark/cocarry-v1/ref-script-2026-07-05/*' --local-dir hosted-download
path=/home/reviewer/work/concerto/hosted-download
$ uv run chamber-eval verify hosted-download/benchmark/cocarry-v1/ref-script-2026-07-05
warning: `VIRTUAL_ENV=/home/reviewer/work/bootstrap-venv` does not match the project environment path `.venv` and will be ignored; use `--active` to target the active environment instead
PASS  bundle:schema                                        v3 ResultBundle
PASS  bundle:eligibility                                   clean working tree
PASS  manifest:REPRO.txt                                   sha256 match
PASS  manifest:episodes_seed0.jsonl                        sha256 match
PASS  manifest:episodes_seed1.jsonl                        sha256 match
PASS  manifest:episodes_seed2.jsonl                        sha256 match
PASS  manifest:episodes_seed3.jsonl                        sha256 match
PASS  manifest:episodes_seed4.jsonl                        sha256 match
PASS  manifest:partners.json                               sha256 match
PASS  manifest:prereg.yaml                                 sha256 match
PASS  manifest:membership                                  no unmanifested files
PASS  sha256sums:REPRO.txt                                 sha256 match
PASS  sha256sums:bundle.json                               sha256 match
PASS  sha256sums:episodes_seed0.jsonl                      sha256 match
PASS  sha256sums:episodes_seed1.jsonl                      sha256 match
PASS  sha256sums:episodes_seed2.jsonl                      sha256 match
PASS  sha256sums:episodes_seed3.jsonl                      sha256 match
PASS  sha256sums:episodes_seed4.jsonl                      sha256 match
PASS  sha256sums:partners.json                             sha256 match
PASS  sha256sums:prereg.yaml                               sha256 match
PASS  summary:n_episodes                                   1750 episodes (schedule expects 1750)
PASS  summary:success_mean                                 recomputed 1.000000 vs stated 1.000000
PASS  summary:success_iqm                                  recomputed 1.000000 vs stated 1.000000
PASS  summary:success_ci_low                               recomputed 1.000000 vs stated 1.000000
PASS  summary:success_ci_high                              recomputed 1.000000 vs stated 1.000000
PASS  partner:ego:ref_script_cocarry_impedance:registered  cocarry_impedance
PASS  partner:imp_stiff_low:registered                     cocarry_impedance
PASS  partner:imp_stiff_high:registered                    cocarry_impedance
PASS  partner:imp_damp_low:registered                      cocarry_impedance
PASS  partner:imp_damp_high:registered                     cocarry_impedance
PASS  partner:imp_lag_bounded:registered                   cocarry_impedance
PASS  partner:imp_blend_b:registered                       cocarry_impedance
PASS  partner:imp_blend_c:registered                       cocarry_impedance
PASS  partner:hashes                                       identity hashes match
PASS  prereg:tag                                           tag prereg-cocarry-baselines-v1-rev2-2026-07-05 exists
PASS  prereg:blob                                          bundle copy matches verified blob SHA
verify: PASS (36/36 checks)

# All steps passed.
```
