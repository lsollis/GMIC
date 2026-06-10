# FedProx salvage: reconstruct per-round globals, score test, compute paper stats

Salvage an interrupted run with no resume/retraining. A test-only federated job reconstructs the
shared global at chosen rounds (train-size-weighted aggregation of the per-site SENT weights),
scores val+test, dumps predictions, and **computes the paper stats during the run** — all logged to
the **server**, whose log persists past job deletion (see "Where the output lands" below):

- **Per-site** rows (breast-level AUC + DeLong CI, Youden(val)→test sens/spec + Wilson CIs) for every
  site and both model classes — `[salvage-site] method=... src_round=... site=...`.
- **Pooled** (the one endpoint needing cross-site fusion) for the shared global —
  `[salvage-pooled] ... src_round=...`.

Each client computes its own breast-level (prob, label) and ships it to the server, which logs both
the per-site and pooled rows centrally — so you get HPU's numbers without touching the HPU node. The
clients also log their own `[salvage-stats]` rows locally (convenience). `tools/salvage_stats.py`
reproduces the identical numbers offline from gathered CSVs. Estimators are shared
(`gmic_job_hpu/app/custom/salvage_metrics.py`); the server per-site/pooled paths are proven equal to
the offline ones.

## Where the output lands (and why it survives job deletion)

The NVFlare job workspace is deleted when the job finishes, so the salvage stats are logged on the
**server**, whose log is redirected to a mounted path:
- `NVFL_LOG_ROOT=/workspace/server_logs` → `<server-host-dir>/server_logs/log.txt`
  (site_folders/server/docker-compose.yml) — NVFlare's `log.txt`, includes every component log line.
- `<server-host-dir>/server_logs/server_console.log` — raw `docker logs` tee
  (site_folders/server/run_server.sh).
- `<server-host-dir>/server_logs/salvage_stats.jsonl` — durable machine-readable copy, one JSON row
  per result (written by the aggregator's `stats_out_dir=/workspace/server_logs`).

To confirm this capture works on a PRIOR run: check that `<server-host-dir>/server_logs/log.txt`
exists and contains component output (e.g. aggregation / `IntimeModelSelector` lines). If it does, the
salvage `[salvage-site]`/`[salvage-pooled]` lines will land there too.

Client-side: prediction CSVs, checkpoints, and `global_trajectory/` persist in each node's
`results_dir` (`/workspace/data/processed/<run>/`). Client `self.log_info` lines (incl. the local
`[salvage-stats]`) go to the transient NVFlare client workspace unless that node tees its console —
the committed HPU/Moffitt containers do not, which is exactly why the stats are routed to the server.

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

Submit the `gmic_job_hpu` job. Read everything from the persistent SERVER log:

```
grep "\[salvage-site\]"   <server-host-dir>/server_logs/log.txt   # per-site rows, all sites, both models
grep "\[salvage-pooled\]" <server-host-dir>/server_logs/log.txt   # pooled rows
cat                       <server-host-dir>/server_logs/salvage_stats.jsonl   # machine-readable copy
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
