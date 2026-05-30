# GMIC Federated Methods — implementation & restricted-machine handoff

One configurable executor + aggregation path supporting six training regimes, switched by
a single `method` flag in the client job config (the λ sweep launches by editing config,
not code). `method ∈ {local, fedavg, fedprox, fedbn, ditto, ditto_modulewise}`, default
`fedavg`. The `gmic_job_hpu/` job carries a **byte-identical** executor + primitives (only
its configs differ: single-GPU, short rounds).

## Files
- `app/custom/fl_utils.py` — primitives (pure torch): `module_group`, `bn_state_keys`,
  `load_global_into`, `proximal_penalty`, `clone_detached_state`, `tolerant_load_pretrained`,
  `gmic_outputs`, `malignant_score`, `gmic_malignant_loss`.
- `app/custom/model/gmic.py` — `forward()` returns `(fusion_logits, y_global, y_local, saliency)`.
- `app/custom/bc_executor.py` — method switch, pretrained init, GMIC loss, Ditto, dumps.
- `app/custom/train/training_core.py` — tuple-aware `evaluate_model` (malignant head).
- `app/config/config_fed_client.json` — `method` block.
- `app/custom/test_fl_methods.py`, `test_executor_integration.py` — CPU tests.

## Model / loss (verified against upstream nyukat/GMIC + the GMIC papers)
- Param→group (module-wise Ditto): `global = ds_net + left_postprocess_net`,
  `local = dn_resnet`, `fusion = mil_attn_V/U/w + classifier_linear + fusion_dnn`
  (135 params, 0 unassigned). BatchNorm = 41 modules / 205 keys, all in the two backbones.
- **Loss = native GMIC deep supervision, malignant head only** (data carries a single
  binary cancer-vs-else label; benign ≡ ¬malignant):
  `BCEWithLogits(fusion[:,1]) + BCE(y_global[:,1]) + BCE(y_local[:,1]) + λ·Σ|saliency[:,1]|`.
  Independent sigmoids, never softmax. Global/local are post-sigmoid probabilities → `BCE`;
  fusion is a raw logit → `BCEWithLogits`. **The loss is computed OUTSIDE `autocast`** (fp32);
  `malignant_score` casts to fp32 before `sigmoid`/`.numpy()` (bf16/fp16 safety under AMP).
  Class imbalance: a single per-sample `pos_weight` (default 3) applied identically to all
  three heads. Benign head (index 0) is present-but-unsupervised so `sample_model_5.p` loads 1:1.
- `malignant = index 1` everywhere (run_model.py: `benign,malignant = pred[0,0],pred[0,1]`).
- Send convention: FULL weights; server `nvflare...FedAvg` controller aggregates by
  NUM_STEPS — one aggregator serves all methods, no server change.

## Per-method behavior
| method | global load | extra train | sent | deployed/eval |
|---|---|---|---|---|
| `local` | skipped (pretrained init) | — | w (ignored) | w |
| `fedavg` | full | — | w | w |
| `fedprox` | full | `+(mu/2)‖w−w_g‖²` | w | w |
| `fedbn` | full, **skip BN keys** | — | w | w (shared + local BN) |
| `ditto` | full (w) | personal pass trains `v`: `loss(v)+prox(v,w_ref,λ)` | w | **v** |
| `ditto_modulewise` | full (w); BN-skip on w if `use_fedbn` | per-group λ personal pass | w | **v** |

## Config schema (client args)
`method, fedprox_mu(0.01), ditto_lambda(0.1), lambda_global(1.0), lambda_local(0.1),
lambda_fusion(null→local), use_fedbn(false), local_epochs(null→epochs), pos_weight(3.0),
loss("gmic"), pretrained_weights_path, lambda_l1(1e-5)`. percent_t is set **only** via
`percent_t_key` (→ PERCENT_T_DICT; "1"→0.02); a disagreeing `gmic_parameters.percent_t`
now raises (single source of truth). λ grid: `lambda_global∈{1,5,10} × lambda_local∈{0.01,0.1,0.5}`.

## Outputs for offline analysis (per site, in `results_dir`)
- `{site}_predictions_{tag}_round{R}_{val,test}.csv` — columns
  `site_id, round, method, split, exam_id, view, path, prob_malignant, label`
  (`tag` = `pretrained_baseline` at round 0, else the method). `prob_malignant` = fp32 sigmoid.
- `{site}_saliency_{tag}_round{R}_{val,test}.npz` — `saliency` (N,h,w fp32, malignant channel,
  post-sigmoid), `path`, `label`, `site_id`, `method`, `split`, `round`.
- **Join key = `path`**, identical in the CSV and the NPZ → map↔prediction↔label↔site↔method
  joins offline with no model. AUC/DeLong CIs from the CSV; paired saliency stats from the NPZ.
- Per-round `[per-site-metric] site=… round=… method=… deployed_val_auc=…` log line.
  Degenerate (zero-positive) eval splits report AUC=NaN (+`n_pos`), excluded from ranking.

## Clone environment (for running the tests here)
Not installed by default. Install on demand: CPU `torch`, `nvflare`, `opencv-python-headless`,
`imageio`, and **pin `numpy<2`** (an opencv install pulls numpy 2.x which breaks anaconda
scipy/sklearn). NVFLARE imports the Unix-only `resource` module — `test_executor_integration.py`
injects a `resource` stub so it runs on Windows.
Tested: `test_fl_methods.py` 8/8, `test_executor_integration.py` 4/4 (incl. a `use_amp=True` round).

## RUN ORDER on the restricted machine (air-gapped; needs real data/GPU)
1. **Cold-parity gate (first).** Load `sample_model_5.p`, run inference on a few real
   preprocessed images per site, confirm `sigmoid(fusion)[:,1]` ranks cancers > controls.
   If near-random, preprocessing diverges from GMIC's pipeline (2944×1920, breast crop,
   per-image zero-mean/unit-std) — fix before anything else. (The executor also runs a
   round-0 `pretrained_baseline` dump automatically.)
2. **Centralized baseline.** `method=local`, **one client on the pooled CSV, `num_rounds=1`,
   `epochs≈40` + early stopping** — must be a single continuous loop (NOT `num_rounds=40,
   epochs=1`, which would trip the cross-round early-stopper reload; see C2). This is the ceiling.
3. **Federated baselines:** `fedavg`, then `fedprox`, `fedbn`.
4. **λ sweep:** `ditto` / `ditto_modulewise` over the grid (short rounds to rank, finish the
   winner at full length).
Collect `predictions_*.csv` + `saliency_*.npz` per site/round/method for offline analysis.
Also: populate the site configs (paths/site IDs — placeholders here) and confirm the label
contract (the loader now hard-asserts strictly {0,1}, breast-level view agreement, on load).

## OPEN — needs your decision before multi-round FL (C2, not yet changed)
The early stopper persists across rounds, so `execute()` reloads the **all-time-best** `w`
(possibly from a prior round) before packaging the FedAvg update — wrong for the global model
(FedAvg expects the current round's local weights); `v` (Ditto) is unaffected. Pending choice:
(A) reset the stopper per round [recommended] or (B) drop the cross-round reload for `w`.
Running centralized as `num_rounds=1` (step 2) sidesteps it.

## Commit history (branch `main`)
`d9d27f5` federated methods · `4e2824f` GMIC loss · `29caeed` saliency dumps + softmax cleanup ·
(this) QC fixes: AMP loss/score fp32, label-contract assert + converter reconcile, image-load
skip, degenerate-AUC NaN, raise_on_partial, percent_t single-source.
