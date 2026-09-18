# Personalized Federated Learning for Equitable Breast Cancer Detection

A federated-learning (FL) training and evaluation system for breast cancer
detection across multiple clinical sites, built on the
[GMIC](https://github.com/nyukat/GMIC) mammography model and
[NVIDIA FLARE](https://github.com/NVIDIA/NVFlare) (NVFLARE).

This code accompanies:

> Sollis LJ, Young PM, Bunnell A, Quon B, Hernandez BY, Wolfgruber TK, Shepherd J.
> **"Personalized Federated Learning for Equitable Breast Cancer Detection in
> Underrepresented Pacific Islander Populations."** MICCAI 2026 Workshop on
> Distributed, Collaborative, and Federated Learning (DeCaF); to appear in
> Springer LNCS.

> **Built on GMIC (NYU).** The underlying image model — the Globally-Aware
> Multiple-instance Classifier — and its preprocessing are the work of Shen et
> al. (NYU); see [`GMIC_MODEL_README.md`](GMIC_MODEL_README.md), the
> [original repository](https://github.com/nyukat/GMIC), and
> [arXiv:2002.07613](https://arxiv.org/abs/2002.07613). This repository extends
> that model with a federated training/evaluation system. It is a derivative
> work and, like GMIC, is licensed under **GNU AGPLv3** (see
> [LICENSE](LICENSE) and [NOTICE](NOTICE)).

---

## What this adds to GMIC

GMIC is a single-model image classifier. This repository wraps it in an NVFLARE
executor so several clinical sites can train a shared model **without pooling
patient data**, and adds the pieces needed to study cross-site and
cross-demographic equity:

- **Federated methods** — FedAvg, FedProx, FedBN, and personalized **Ditto**
  (both a scalar proximal weight and a module-wise variant with separate
  weights for the model's global / local / fusion blocks).
- **Per-round evaluation** — each round dumps per-site validation/test
  predictions so AUC, DeLong CIs, and operating points can be computed offline
  and pooled across sites.
- **Crash-resume** — an interrupted federated run can resume from the last
  completed round without restarting.
- **Constrained-GPU training** — optional per-site precision and micro-batch
  controls that let a memory-limited GPU participate (see below).
- **Subgroup fairness analysis** — post-hoc per-race/ethnicity metrics
  (AUC + CIs, sensitivity at fixed specificity, etc.) for any deployed model.

## Repository layout

| Path | Purpose |
|------|---------|
| `gmic_job_hpu/` | Real (multi-site) federated job; the executor lives at `app/custom/bc_executor.py`. |
| `gmic_job/` | Base federated job (single canonical executor, shared by all jobs). |
| `gmic_job_ditto_sim/`, `gmic_job_ditto_mw_sim/`, `gmic_job_fedprox_sim/`, `gmic_job_fedbn_sim/` | NVFLARE **simulator** jobs, one per method, for local multi-site experiments. |
| `pool_report_job/` | Pools each site's per-round predictions into combined AUC/DeLong/operating-point statistics. |
| `ditto_sweep/` | Hyperparameter sweeps (Ditto λ; FedProx μ). |
| `subgroup_fairness.py` | Per-race/ethnicity fairness for one site's deployed model. |
| `run_all_subgroups.py` | Runs `subgroup_fairness.py` across every method and builds a combined table. |
| `dump_ditto_perround_preds.py` | Dumps per-round predictions for personalized (Ditto) runs. |
| `tools/` | Operational helpers (salvage/resume runbooks). |
| `site_folders/` | Deployment templates: NVFLARE server/client Docker kits and a CSV→GMIC converter. |
| `gmic-localhost.yml`, `master_template.yml` | NVFLARE provisioning: project spec + workspace template. |

All six job folders carry a **byte-identical** copy of `bc_executor.py` (NVFLARE
requires per-job custom code); change one and re-copy to keep them in sync.

## Requirements & setup

The model, data pipeline, and FL stack run in a Docker container. See
[`GMIC_MODEL_README.md`](GMIC_MODEL_README.md) for the model/preprocessing
prerequisites and `Dockerfile` / `docker-compose.yml` for the container. In
brief: PyTorch + NVFLARE, one GPU per site, mammography images preprocessed to
2944×1920 16-bit PNGs.

Each site provides a metadata CSV (see
[`site_folders/sample_gmic_data_format.csv`](site_folders/sample_gmic_data_format.csv)
for the schema: `patient_id, exam_id, laterality, view, file_path,
exam_level_label, view_level_label, split_group, ...`). No patient data is
included in this repository.

The GMIC pretrained weights (the FL warm-start, `sample_model_1..5.p`) are **not**
shipped here — download them from the [GMIC repository](https://github.com/nyukat/GMIC)
and place them where your config expects (default `/workspace/models/`).

## Deploying the federated system

Deployment has two layers. NVFLARE provisioning generates each participant's FL
startup kit; a small Docker template wraps it into a runnable container.

1. **Provision the startup kits.** `gmic-localhost.yml` is the NVFLARE project
   spec (server, clients, admin) and `master_template.yml` is the standard
   NVFLARE workspace template. Edit the participant list to your sites, then:

   ```bash
   nvflare provision -p gmic-localhost.yml
   ```

   This writes a startup kit per participant (certificates, `fed_server.json` /
   `fed_client.json`, `start.sh`, `sub_start.sh`, `docker.sh`).

2. **Wrap each kit in a container.** `site_folders/server/` and
   `site_folders/client/` are **templates** — a `Dockerfile`, a
   `docker-compose.yml`, and a run script — for the server and for a client.
   Copy the matching template alongside a participant's startup kit, set the
   placeholders (`ORG_NAME` for the server; `SITE_NAME` for a client, matching
   the provisioned participant name), and run `run_server.sh` / `run_client.sh`.
   `site_folders/csv_to_gmic_converter.py` helps convert a site's registry into
   the expected metadata schema.

> These are templates, not our production kits: a deployer provisions their own
> project (their hosts, their certificates) rather than reusing ours.

## Configuring a federated method

A job's method and hyperparameters are set in `app/config/config_fed_client.json`
(executor args). Key knobs:

| Config key | Meaning |
|------------|---------|
| `method` | `fedavg`, `fedprox`, `fedbn`, `ditto`, or `ditto_modulewise`. |
| `fedprox_mu` | FedProx proximal strength (μ). |
| `ditto_lambda` | Ditto proximal weight (scalar Ditto). |
| `lambda_global` / `lambda_local` / `lambda_fusion` | Per-block Ditto weights (module-wise). |
| `use_fedbn` | Keep BatchNorm layers local (FedBN). |
| `use_amp` | Mixed-precision training. |
| `resume_from_local_round` | Resume an interrupted run from this round (`-1` = fresh). |

Per-folder `FEDERATED_METHODS.md` files document each method in detail.

## Training on a memory-constrained GPU

The federated methods run best on ample-memory GPUs, but a site on a smaller
card (e.g. a 16 GiB RTX A4000) can still participate using the optional,
**default-off** toggles below. They are resolved from the FL identity at
runtime, so one config deployed to every site affects only the listed site — no
per-site app copies, and no need to move a site to a larger GPU. Leave them
unset on capable GPUs; every default reproduces standard behavior.

| Config key (executor arg) | Effect when set |
|---------------------------|-----------------|
| `personal_amp_by_site` | Per-site precision for the Ditto personal pass, e.g. `{"SITE_X": false}` forces fp32 there while others keep AMP. Unset → follows `use_amp`. |
| `personal_batch_size_by_site` | Per-site micro-batch for the personal pass, e.g. `{"SITE_X": 8}`; chunks accumulate to the full effective batch (only BatchNorm sees the chunk). Unset → no chunking. |
| `train_batch_size_by_site` | Same, for the main (shared-weight) training pass. |
| `heartbeat_interval_s` | Seconds between watchdog-thread progress logs during long phases; logs only, never aborts. `0` (default) disables. |
| `memory_efficient` | Release cached CUDA memory between passes. |
| `stage_sync`, `debug_devices` | Diagnostics for locating a stalled/faulting CUDA op. |

Micro-batch chunking is gradient-exact (the optimizer sees the full effective
batch); only BatchNorm statistics are computed on the smaller chunk.

## Analysis tools

- **Pooled statistics** — submit `pool_report_job` to combine per-site
  predictions into AUC, DeLong CIs, Youden thresholds, and operating-point
  metrics.
- **Subgroup fairness** — after a run, compute per-race/ethnicity metrics for a
  site's deployed model. The site label and the race/ethnicity columns and
  code→group mapping are all runtime parameters, so the tool carries no
  site-specific schema:

  ```bash
  python subgroup_fairness.py \
      --pred  <SITE>_predictions_<method>_round<N>_test.csv \
      --val   <SITE>_predictions_<method>_round<N>_val.csv \
      --meta  <site_registry>.csv \
      --site  <SITE> --eth-col <race_column> --eth-map <map.json>
  ```

  `run_all_subgroups.py` runs this across every method (auto-selecting each
  method's best-validation round) and writes a combined table plus a
  group × method summary.

## Citation

If you use this code, please cite both the federated-learning paper (above) and
the original GMIC work:

```bibtex
@article{shen2021gmic,
  title   = {An interpretable classifier for high-resolution breast cancer
             screening images utilizing weakly supervised localization},
  author  = {Shen, Yiqiu and Wu, Nan and Phang, Jason and Park, Jungkyu and
             Liu, Kangning and Tyagi, Sudarshini and Heacock, Laura and
             Kim, S. Gene and Moy, Linda and Cho, Kyunghyun and Geras, Krzysztof J.},
  journal = {Medical Image Analysis},
  year    = {2021}
}
```

## License

GNU Affero General Public License v3.0 (AGPLv3). This is a derivative of GMIC
(© 2020 the GMIC authors, NYU) and remains under the same license; the
federated-learning additions are © 2026 Shepherd Research Lab, University of
Hawaiʻi Cancer Center. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
