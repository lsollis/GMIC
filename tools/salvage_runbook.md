# FedProx salvage: reconstruct per-round globals, score test, compute paper stats

Salvage an interrupted run with no resume/retraining. Reconstructs the federated global at chosen
rounds (train-size-weighted aggregation of the per-site SENT weights), scores val+test, dumps
predictions, and computes breast-level AUC + DeLong CIs and Youden-selected sens/spec + Wilson CIs,
per site and pooled. Two pieces:

1. **Federated test-only job** — `GMICFederatedExecutor` salvage mode (`bc_executor._salvage_round`).
2. **Offline stats** — `tools/salvage_stats.py` (no model/data/GPU; runs on gathered CSVs).

## What gets produced

Per configured round `N`, each client dumps prediction CSVs into its results dir:
- `{site}_predictions_incoming_global_round{N}_{val,test}.csv` — the **reconstructed shared global**
  built from round-N contributions (same model at every site → per-site **and** pooled endpoints).
- `{site}_predictions_deployed_local_round{N}_{val,test}.csv` — that site's **as-deployed** model at
  round N (a different model per site → **per-site only**, no pooled).

Indexing note: aggregating round-N SENT weights yields the global that *entered round N+1*. Files are
labeled by the source round N; select the pooled-best by each global's own measured **val** AUC.

## 1. Run the federated salvage job

In `gmic_job_hpu/app/config/config_fed_client.json`, add to the executor `args`:

```json
"salvage_eval_rounds": [6, 8, 10, 14, 23, 26]
```

(Shortlist = pooled candidates {6,8,10} ∪ per-site bests {14=HPU, 23=UHCC, 26=last}. Add/remove
rounds freely.) Point `resume_ckpt_dir` at the dir holding each node's checkpoints if not the default
`results_dir`; the executor reads `global_trajectory/{site}_global_round_{N}.pth` (authoritative SENT
weights), falling back to `{site}_gmic_model_round_{N}.pth`.

In `gmic_job_hpu/app/config/config_fed_server.json`, set the controller to **one more than the number
of rounds** (a trailing round is needed to evaluate the last reconstruction):

```json
"num_rounds": 7
```

Use the existing FedAvg controller + `InTimeAccumulateWeightedAggregator` (already configured) — it
weights submissions by `num_examples` (= train_size) exactly as the original run did. No training
happens; each round only loads weights and runs eval, so the HPU-dropout window is seconds, not hours.

**Built-in pipeline check (free):** `deployed_local` val AUC at round N reproduces the original logged
`deployed_val_auc[N]` (epochs=1: sent == best-val deployed). E.g. UHCC `deployed_local` round 23 val
AUC should be ≈ 0.8924. If it matches, the data splits / loading / eval are faithful end-to-end.

## 2. Gather + analyze

Copy every `*_predictions_*.csv` from all three nodes into one directory, then:

```bash
python tools/salvage_stats.py --pred-dir <gathered_csvs> --out-prefix results/fedprox_salvage
```

Outputs `<prefix>_metrics.csv` and `<prefix>_table.md`. Per (method, round) it reports, per endpoint
(each site; pooled for `incoming_global` only): n breasts/positives, **val AUC** (round selection),
**test AUC + DeLong 95% CI**, **Youden threshold** (from that endpoint's val), and **sens/spec with
Wilson 95% CIs** (on test). Pick the **pooled-best `incoming_global` round** by pooled val AUC; report
its pooled test row (the FedProx bracket baseline) + its per-site rows (heterogeneity panel). For the
per-site personalization view, use each site's own best round under `deployed_local` and/or
`incoming_global`.

Validate the estimators any time with `python tools/salvage_stats.py --selftest`.
