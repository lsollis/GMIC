# GMIC Federated Methods — implementation & handoff

Single configurable executor + aggregation path supporting six regimes, switched
by **one config flag** (`method`) in the client job config. No per-method code
copies; the λ sweep launches by editing config only.

`method ∈ {local, fedavg, fedprox, fedbn, ditto, ditto_modulewise}` — default
`fedavg` (reproduces prior behavior).

## Files
- `app/custom/fl_utils.py` — **new.** Pure-torch primitives (no NVFLARE import):
  `module_group`, `assert_grouping_complete`, `bn_state_keys`, `load_global_into`,
  `proximal_penalty`, `clone_detached_state`, `tolerant_load_pretrained`.
- `app/custom/bc_executor.py` — **edited.** Config args, pretrained init in
  `initialize()`, method-aware `execute()`, FedProx term in `_local_train`, and
  new methods `_consume_global`, `_train_personal_model`, `_dump_baseline`,
  `_dump_predictions`.
- `app/config/config_fed_client.json` — **edited.** Adds the `method` block.
- `app/custom/test_fl_methods.py`, `app/custom/test_executor_integration.py` —
  **new.** CPU synthetic tests.

> The `gmic_job_hpu/` job has a **byte-identical** executor; mirror the same three
> file changes there. Only its configs differ (single-GPU, 3-round smoke).

## Verified facts (from this clone)
- **Parameter → group mapping** (confirmed by importing the model; 0 unassigned,
  135 learnable params: global=69, local=60, fusion=6):
  - `global` = `ds_net.*`, `left_postprocess_net.*`
  - `local`  = `dn_resnet.*`
  - `fusion` = `mil_attn_V/U/w.*`, `classifier_linear.*`, `fusion_dnn.*`
- **BatchNorm**: 41 BN modules → **205 BN state_dict keys**, all in the two
  backbones (21 in `ds_net`, 20 in `dn_resnet`). FedBN skips exactly these on load.
- **Pretrained checkpoint** `sample_model_5.p`: loads **258/259** keys with names
  matching 1:1 (no `module.`/remap). Only benign mismatches: missing `_device_ref`
  (empty buffer), unexpected `shared_rep_filter.weight` (other GMIC variant).
- **Send convention**: FULL weights (`ParamsType.FULL`), aggregated as weighted
  FedAvg by `MetaKey.NUM_STEPS_CURRENT_ROUND` under the existing
  `nvflare.app_common.workflows.fedavg.FedAvg` controller. **No server-side change
  needed** — one aggregator handles all methods (FedBN uses skip-on-load, not
  payload trimming, so the payload is always complete).

## Decisions locked with the user
1. Grouping as above (confirmed).
2. **Always init from pretrained** — `initialize()` tolerant-loads the checkpoint
   into every client, independent of the server persistor. Makes the round-0
   baseline meaningful and `local` start pretrained.
3. Checkpoint = `sample_model_5.p` (single model).
4. `ditto_modulewise` + `use_fedbn`: FedBN BN-skip applies **only to `w`**, not `v`.

## Per-method behavior (as implemented)
| method | global load | extra train | sent | deployed/eval |
|---|---|---|---|---|
| `local` | **skipped** every round (pretrained init from `initialize()`) | — | w (ignored next round) | w |
| `fedavg` | full | — | w | w |
| `fedprox` | full | `+ (mu/2)‖w−w_global‖²` on w-pass | w | w |
| `fedbn` | full **skip BN keys** | — | w (full payload; BN never clobbered on load) | w (shared + local BN) |
| `ditto` | full (w) | personal pass trains `v`: `loss(v)+prox(v,w_ref,λ)` | w | **v** |
| `ditto_modulewise` | full (w); BN-skip on w if `use_fedbn` | personal pass with per-group λ dict | w | **v** |

`v` (Ditto personal model) + its optimizer are executor instance attributes,
initialized **once** from the pretrained `w` and **persisted across rounds**
(never reset to the global). Personal pass runs in fp32 (no AMP).

## Config schema (client job args)
```json
"method": "fedavg",
"fedprox_mu": 0.01,
"ditto_lambda": 0.1,
"lambda_global": 1.0,
"lambda_local": 0.1,
"lambda_fusion": null,          // null → ties to lambda_local
"use_fedbn": false,
"local_epochs": null,          // null → uses existing "epochs"
"pretrained_weights_path": "/workspace/models/sample_model_5.p"
```
λ sweep grid (edit config only): `lambda_global ∈ {1,5,10} × lambda_local ∈
{0.01,0.1,0.5}`, `lambda_fusion` defaults to `lambda_local`. Rank on short runs
(`num_rounds`/`local_epochs`/`epochs` small), then finish the winner at full length.

## Outputs for offline analysis
Per site, written to `results_dir`:
- `{site}_predictions_pretrained_baseline_round0_{val,test}.csv` — cold-start ref.
- `{site}_predictions_{method}_round{R}_{val,test}.csv` — final-round deployed model.
- Columns: `site_id, round, method, split, exam_id, view, path, prob_malignant, label`.
- Per-round, per-site log line: `[per-site-metric] site=… round=… method=… deployed_val_auc=…`.
DeLong CIs and σ(AUC) are computed offline from these dumps (not in the executor).

## RUN ON THE RESTRICTED MACHINE (needs real data / cannot run in the clone)
1. **Actual federated runs** for each method (and the λ sweep) via the NVFLARE job.
2. **Preprocessing-parity check** for the pretrained baseline: the clone's pipeline
   uses 2944×1920 + per-image zero-mean/unit-std + breast crop + chest-wall
   orientation, which **matches GMIC's expected input** — but confirm on real
   images that baseline AUC is in the expected range (not artificially low).
3. **Populate site configs** (paths, site IDs, data locations) — left as
   placeholders here (`/workspace/...`).
4. Confirm the server persistor still points at the checkpoint (it does in
   `gmic_job`); `initialize()` also loads it directly, so baseline is robust either way.

## Tested in the clone (CPU, synthetic)
Torch and NVFLARE are not installed in the clone by default; install CPU torch and
nvflare on demand. NVFLARE imports the Unix-only `resource` module, so on Windows
`test_executor_integration.py` injects a small `resource` stub (no-op on Linux).
- `test_fl_methods.py` (pure torch): grouping, BN-skip, proximal (per-group λ
  exactness), pretrained load (258/259), forward smoke, Ditto persistence —
  **6/6 pass.**
- `test_executor_integration.py` (drives `execute()` per method with a synthetic
  loader + bare FLContext): per-method smoke → valid Shareable + baseline/pred
  dumps; fedavg no-regression; Ditto cross-round persistence (tests 4/5/6).
