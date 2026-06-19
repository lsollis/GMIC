#!/usr/bin/env python3
"""Dump the best-val PERSONAL model's val+test predictions for personalized runs (Ditto family),
so they can be reported best-val-selected -- apples-to-apples with the shared-model methods.

Why: the executor saves the best-val personal model as WEIGHTS only
(<site>_best_val_overall_gmic_model.pth); per-round personal preds are never dumped (only the
final round). This loads that checkpoint and re-dumps its preds by REUSING the executor's own
initialize() + _dump_predictions, so the data splits / preprocessing / model build / breast
schema are identical to the run. Writes <site>_predictions_<out_tag>_round<r>_<split>.csv into
each site's results dir; the stats notebook then reads tag=<out_tag>.

Self-check: recomputes breast-level val AUC from the dumped val preds and compares to the val AUC
the executor recorded in <site>_best_val_overall_gmic_metrics.json. They must match (~1e-2) or the
checkpoint/data wiring is wrong -- it prints OK / CHECK! per site.

Run on the GPU machine (needs torch + data + the built crop caches), from the repo root:
    python dump_bestval_preds.py
"""
import os, sys, json, csv, glob, inspect, logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = "/raid/home/lsollis/GMIC/GMIC"
CLIENTS = ["UHCC", "HPU", "RSNA-GCP"]
# job_dir -> output prediction tag (kept distinct from the final-round 'ditto'/'ditto_modulewise')
JOBS = {
    f"{REPO}/gmic_job_ditto_sim": "ditto_bestval",
    # f"{REPO}/gmic_job_ditto_mw_sim": "ditto_modulewise_bestval",   # uncomment when it finishes
}


class FakeCtx:
    def __init__(self, name):
        self._n = name
    def get_identity_name(self):
        return self._n


def run_job(job_dir, out_tag):
    sys.path.insert(0, os.path.join(job_dir, "app", "custom"))
    from bc_executor import GMICFederatedExecutor
    from fl_utils import breast_aggregate

    cfg = json.load(open(os.path.join(job_dir, "app", "config", "config_fed_client.json")))
    args = cfg["executors"][0]["executor"]["args"]
    sig = inspect.signature(GMICFederatedExecutor.__init__)
    kw = {k: v for k, v in args.items() if k in sig.parameters}

    for site in CLIENTS:
        print(f"\n=== {os.path.basename(job_dir)} / {site} -> tag={out_tag} ===")
        ex = GMICFederatedExecutor(**kw)
        ex._identity = site
        if not getattr(ex, "_logger", None):
            ex._logger = logging.getLogger("infer")
        ex.log_info = lambda *a, **k: None        # bypass NVFlare fl_ctx logging
        ex.log_warning = lambda *a, **k: None
        ex.initialize()                            # builds data_loader + model (reuses crop cache)

        rdir = ex.results_dir
        ckpt = os.path.join(rdir, f"{site}_best_val_overall_gmic_model.pth")
        mjson = os.path.join(rdir, f"{site}_best_val_overall_gmic_metrics.json")
        if not os.path.isfile(ckpt):
            print(f"  [skip] no checkpoint at {ckpt}")
            del ex; continue
        meta = json.load(open(mjson)) if os.path.isfile(mjson) else {}
        rstar = int(meta.get("round", -1))
        rec_auc = meta.get("val_auc", meta.get("auc"))

        model = ex._underlying.to(ex.device).eval()
        sd = torch.load(ckpt, map_location=ex.device)
        sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected (benign if small)")

        fctx = FakeCtx(site)
        for split in ("val", "test"):
            ex._dump_predictions(fctx, split=split, model=model, round_idx=rstar,
                                 method_tag=out_tag, save_saliency=False)

        # self-check on val
        vpaths = glob.glob(os.path.join(rdir, f"{site}_predictions_{out_tag}_round{rstar}_val.csv"))
        if vpaths:
            rows = list(csv.DictReader(open(vpaths[0], newline="")))
            bp, by = breast_aggregate([r["exam_id"] for r in rows], [r["view"] for r in rows],
                                      [float(r["prob_malignant"]) for r in rows], [int(r["label"]) for r in rows])
            chk = roc_auc_score(by, bp) if len(set(by.tolist())) > 1 else float("nan")
            ok = (isinstance(rec_auc, (int, float)) and abs(chk - rec_auc) < 1e-2)
            print(f"  best-val round={rstar}  recorded_val_auc={rec_auc}  recomputed={chk:.4f}  "
                  f"{'OK' if ok else 'CHECK! (mismatch -> do not trust)'}")
        del ex, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    for job_dir, out_tag in JOBS.items():
        run_job(job_dir, out_tag)
    print("\nDone. Point the notebook's Ditto entry at tag '<...>_bestval'.")
