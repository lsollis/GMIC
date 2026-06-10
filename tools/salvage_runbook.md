# FedProx salvage: reconstruct per-round globals, score test, compute paper stats

Salvage an interrupted run with no resume/retraining. A test-only federated job reconstructs the
shared global at chosen rounds (train-size-weighted aggregation of the per-site SENT weights),
scores val+test, dumps predictions, and **computes the paper stats during the run**, written BOTH on
the clients and on the server (two-way, multiple copies):

- **Per-site** rows (breast-level AUC + DeLong CI, Youden(val)→test sens/spec + Wilson CIs) for every
  site and both model classes — `[salvage-site] / [salvage-stats]`.
- **Pooled** (the one endpoint needing cross-site fusion) for the shared global — `[salvage-pooled]`.

`tools/salvage_stats.py` reproduces the identical numbers offline from gathered CSVs. Estimators are
shared (`.../salvage_metrics.py`); the server per-site/pooled paths are proven equal to the offline
ones.

## Where the output lands (multiple copies, two directions)

**On each client** (its `results_dir` = `/workspace/data/processed/<run>/`, persists with the client
job folder):
- `{site}_predictions_{method}_round{N}_{val,test}.csv` — raw breast/view predictions.
- `{site}_salvage_{method}_round{N}_metrics.json` — that site's computed per-site row (durable file).
- `POOLED_salvage_incoming_global_round{N}_metrics.json` — the pooled row, **pushed back from the
  server** so the clients hold it too.
- client log: `[salvage-stats]` (its own per-site) and `[salvage-pooled] (from server)`.

**On the server** (`<server-host-dir>/server_logs/`, mounted via `NVFL_LOG_ROOT=/workspace/server_logs`
so it survives job-workspace deletion — see site_folders/server/docker-compose.yml + run_server.sh):
- `log.txt` and `server_console.log` — `[salvage-site]` (all sites) and `[salvage-pooled]`.
- `salvage_stats.jsonl` — durable machine-readable copy, one JSON row per result.

How the pooled flows back to the clients: the aggregator computes pooled in `aggregate()` and queues
it on `salvage_bus`; `SalvageShareableGenerator` drains the queue into the next round's broadcast
header (the reliable server→client channel), which the client reads in `_salvage_round`. Fully
defensive — if that channel ever fails, pooled still lives on the server and is recomputable offline.

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
- `"num_rounds": 8` — `len(salvage_eval_rounds) + 2` (one trailing round to evaluate the last
  reconstruction, one more so its pooled row flushes back to the clients). **Revert to your training
  value (e.g. 40+) for a normal run.**
- aggregator → `salvage_pooled_aggregator.SalvagePooledAggregator` and shareable_generator →
  `salvage_shareable_generator.SalvageShareableGenerator` — drop-in supersets of the stock components
  (identical aggregation/broadcast, plus the stats side-channels). Safe to leave for normal training
  (dormant when no salvage header is present).

No training happens; each round only loads weights and runs eval, so the HPU-dropout window is
seconds. Weighting is the freshly-preprocessed `train_size` (matches the original aggregation at
epochs=1) — computed automatically per site, nothing to supply.

## Run it

Submit the `gmic_job_hpu` job. Read the results from EITHER side:

```
# clients (per node results_dir; persists with the client job folder)
cat  <run>/{site}_salvage_*_metrics.json          # per-site rows for that node
cat  <run>/POOLED_salvage_*_metrics.json          # pooled rows (pushed back from server)
grep "\[salvage-"  <client log>                   # [salvage-stats] + [salvage-pooled] (from server)

# server (persists past job deletion)
grep "\[salvage-site\]"   <server-host-dir>/server_logs/log.txt   # per-site, all sites, both models
grep "\[salvage-pooled\]" <server-host-dir>/server_logs/log.txt   # pooled rows
cat                       <server-host-dir>/server_logs/salvage_stats.jsonl   # machine-readable
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
