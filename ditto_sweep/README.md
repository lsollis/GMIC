# Ditto λ sweep (NVFLARE simulator, single multi-GPU host)

Finds the Ditto `ditto_lambda` (personalization ↔ pull-to-global trade-off) that maximizes
the **worst-site** personal-model val AUC, by running one simulator job per λ on the DGX.
All other hyperparameters match the federated run (loss=gmic, epochs=1/round, lr 3e-5/3e-7,
freeze 2, balanced sampler, pos_weight 1.0).

## What it does
For each λ in the grid it renders a job (custom code from `--base-job`, configs from
`templates/`), runs `nvflare simulator` with 3 clients (UHCC/HPU/RSNA-GCP) each on its own
GPU, then reads each client's best personal val AUC and ranks λ by worst-site.

- **Per-client isolation** on the shared filesystem: executor `data_path_map` (CSVs),
  `preprocess_cache_dir_map` (crops), and `{site}` path substitution.
- **Crop cache** is fixed per site and shared across all λ runs → preprocessing happens
  once. A 1-round **warmup** builds caches before any parallel run (auto-skipped if present).
- **Concurrency**: up to one running job per GPU pool (default 2 pools × 3 GPUs).

## Prerequisites on the DGX
1. **All three sites' data present locally**, at the paths in
   `templates/config_fed_client.json` → `data_path_map` (CSVs) and the full image paths
   *inside* those CSVs (`file_path` column). The DGX must hold UHCC + HPU + RSNA images.
2. `sample_model_5.p` at `/workspace/models/sample_model_5.p` (fail-loud if missing).
3. `nvflare` on PATH; GPUs 1–6 free (GPU 0 is the live FL run).
4. Edit `templates/config_fed_client.json` if your CSV paths or cache locations differ from
   the defaults (`/workspace/data/...`, `/workspace/data/processed/ditto_sim_cache/<site>`).

## Run
```bash
cd ditto_sweep
python launcher.py \
  --base-job ../gmic_job \
  --lambdas 0.01,0.05,0.1,0.5,1.0,2.0 \
  --rounds 20 \
  --gpu-pools "1,2,3;4,5,6" \
  --output-base /workspace/sim/ditto \
  --work-root  /workspace/sim/ditto_runs
```
- `--dry-run` renders jobs and prints the simulator commands without launching.
- `--skip-warmup` if the caches are already built.
- `--metric mean_site` to rank by mean instead of worst-site (worst-site is the default).

## Output
- Live per-run logs: `<work-root>/run_l<λ>.log`, `<work-root>/warmup.log`.
- Per-site results: `<output-base>/l<λ>/<site>/<site>_best_val_overall_gmic_metrics.json`.
- Final ranked table is printed and saved to `<work-root>/sweep_summary.json`:
  ```
   lambda  worst_site   mean_site         UHCC         HPU    RSNA-GCP
      0.1      0.8420      0.8710       0.8830      0.8420      0.8880  <- BEST
      ...
  BEST (worst_site): lambda=0.1  worst_site=0.8420
  ```

## Notes / knobs
- **GPUs**: `--gpu-pools "1,2,3;4,5,6"` = two concurrent λ's, three clients each. Use a
  single pool (`"1,2,3"`) to run strictly sequentially, or `--threads 1` to round-robin
  clients on fewer GPUs.
- **Variant**: this sweeps plain `method=ditto` (single λ). For per-module λ's switch to
  `ditto_modulewise` (`lambda_global/local/fusion`) — different search, not wired here.
- **Final model**: the sweep selects λ by val AUC. Re-run the winning λ (or evaluate its
  saved `*_best_val_overall` personal models on the test split) for reported test metrics.
- Cannot be tested on the Windows clone (no GPU/data); run on the DGX.
