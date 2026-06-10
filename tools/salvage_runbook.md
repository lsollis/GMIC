# FedProx salvage: reconstruct per-round globals, score test, compute paper stats

Salvage an interrupted run with no resume/retraining. A test-only federated job reconstructs the
shared global at chosen rounds (train-size-weighted aggregation of the per-site SENT weights),
scores val+test, dumps predictions, and **computes the paper stats during the run**:

- **Per-site** rows (breast-level AUC + DeLong CI, Youden(val)→test sens/spec + Wilson CIs) are
  computed at each node and printed to that node's log — so you get every site, including ones you
  can't pull files from. Grep `[salvage-stats]`.
- **Pooled** (the one endpoint needing cross-site fusion) is computed on the **server** and printed
  to the server log. Grep `[salvage-pooled]`.

`tools/salvage_stats.py` reproduces the identical numbers offline from gathered CSVs (verification, or
when you prefer files to logs). Estimators are shared (`gmic_job_hpu/app/custom/salvage_metrics.py`)
and the server-pooled path is proven equal to the offline pooled.

## What gets evaluated, per configured round N

- `incoming_global` — the **reconstructed shared global** built from round-N contributions (same model
  at every site → per-site **and** pooled). Indexing: aggregating round-N SENT weights yields the
  global that *entered round N+1*; files/logs are labeled by source round N. Select the pooled-best by
  each global's own **val** AUC (logged as `valAUC=`).
- `deployed_local` — each site's **as-deployed** model at round N (a different model per site →
  per-site only, never pooled).

## Config (already set in `gmic_job_hpu`)

`config_fed_client.json` (executor args):
- `"salvage_eval_rounds": [6, 8, 10, 14, 23, 26]` — shortlist = pooled candidates {6,8,10} ∪ per-site
  bests {14 HPU, 23 UHCC, 26 last}. Empty/absent ⇒ normal training (salvage off).
- `"salvage_ckpt_dir": "/workspace/data/processed/fedprox_3client_fixedinit_20260609"` — the
  interrupted run's dir (holds `global_trajectory/{site}_global_round_{N}.pth`, the authoritative
  SENT weights; falls back to `{site}_gmic_model_round_{N}.pth`). Defaults to `results_dir` if unset.

`config_fed_server.json`:
- `"num_rounds": 7` — **one more** than `len(salvage_eval_rounds)` (trailing round evaluates the last
  reconstruction). **Revert to your training value (e.g. 40+) for a normal run.**
- aggregator swapped to `salvage_pooled_aggregator.SalvagePooledAggregator` — a drop-in superset of
  `InTimeAccumulateWeightedAggregator`: identical weighted aggregation, plus the pooled side-channel.
  Safe to leave in place for normal training (dormant when no salvage header is sent).

No training happens; each round only loads weights and runs eval, so the HPU-dropout window is
seconds. Weighting is the freshly-preprocessed `train_size` (matches the original aggregation at
epochs=1) — computed automatically per site, nothing to supply.

## Run it

Submit the `gmic_job_hpu` job. Read the results from the logs:

```
grep "\[salvage-stats\]"  <each client log>     # per-site rows (all sites)
grep "\[salvage-pooled\]" <server log>          # pooled rows
```

**Built-in pipeline check (free):** `deployed_local` val AUC at round N reproduces the original logged
`deployed_val_auc[N]` (epochs=1: sent == best-val deployed). E.g. UHCC `deployed_local` src_round=23
val AUC ≈ 0.8924. If it matches, splits/loading/eval are faithful end-to-end.

## Offline alternative / verification

If you can gather every `*_predictions_*.csv` from the nodes into one directory:

```bash
python tools/salvage_stats.py --pred-dir <gathered_csvs> --out-prefix results/fedprox_salvage
```

Writes `<prefix>_metrics.csv` and `<prefix>_table.md`. Validate estimators with `--selftest`.

## Reporting

Pick the **pooled-best `incoming_global` round** by pooled val AUC; report its pooled test row (the
FedProx bracket baseline) + per-site rows (heterogeneity panel). For the per-site personalization
view, use each site's own best round under `deployed_local` and/or `incoming_global`.

## Note on data sharing

Each client ships its de-identified breast-level (prob, label) for the shared global to the server so
pooled can be computed (no images, no IDs — the same scores already written to the local CSVs). If
your governance disallows even this, remove the aggregator swap and compute pooled offline instead.
