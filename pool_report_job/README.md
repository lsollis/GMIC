# pool_report_job — pooled salvage stats via the FL channel (no node file access)

Computes the **POOLED** breast-level AUC (DeLong CI) + Youden(val)→test sens/spec (Wilson CI) across
all sites for the salvage `incoming_global` predictions, and logs it to the **server** log — so you can
read it through the admin console even for a node whose files you can't reach (e.g. HPU).

It's a one-round, no-GPU job: each client reads the prediction CSVs the salvage already wrote to its
`results_dir`, breast-aggregates, and ships the de-identified `(prob, label)` arrays; the server
concatenates across sites, pools, and logs `[salvage-pooled]`. (This works where the earlier
server-side attempt didn't, because a custom `ModelController` receives the client results directly —
the stock `FedAvg` workflow ignored the `aggregator` component.)

## Run it

Submit to the SAME federation that ran the salvage (server + the 3 site clients), via the admin
console — e.g. `submit_job pool_report_job`. One round, a few seconds per client, no GPU.

Set the path if your salvage `results_dir` differs (it's in `config_fed_client.json` →
`results_dir`, default `/workspace/data/processed/fedprox_3client_fixedinit_20260609`). The executor
globs `{site}_predictions_incoming_global_round*_{val,test}.csv` there.

## Read the result

On the server (admin console), grep `log.txt`:

```
salvage-pooled
```

One line per `(method, round)`:
`[salvage-pooled] method=incoming_global round=26 POOLED n=... (pos=...) valAUC=...  testAUC=... [CI]  thr=...  Sens=... [CI]  Spec=... [CI]`

Also written to `/workspace/server_logs/salvage_pooled.json` (the server config's `out_file`).

## Pick the round to report

Choose the `incoming_global` round with the highest **pooled valAUC** → report its pooled **test** AUC
+ DeLong CI as the FedProx baseline. Sanity: each pooled `n` should be ~the sum of the three sites'
test breasts (≈ 878 + 396 + 396 = 1670 for test, ~826 for val).

## Notes

- `deployed_local` is deliberately excluded (a different model per site → not poolable).
- The pooled math is identical to `tools/salvage_stats.py`; verified equal on synthetic 3-site data.
- Needs all three sites online for the one round (seconds, no GPU — minimal HPU exposure).
