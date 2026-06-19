#!/usr/bin/env python3
"""Dump per-ROUND val+test predictions for personalized runs (Ditto family), so they get a full
per-round trajectory -- exactly like the shared methods' `incoming_global` -- and can be reported
best-val-selected (per-site AND pooled), fully apples-to-apples.

The run saves the deployed PERSONAL model every round as weights
(<site>_gmic_model_round_{r}.pth, via _save_local_model) but never dumps its per-round preds.
This loads each round's checkpoint and dumps its val+test preds by REUSING the executor's own
initialize() + _dump_predictions (identical splits / preprocessing / model build / breast schema),
tagged <out_tag> (default 'ditto_perround'). The data loader is built ONCE per site and reused
across all rounds (only the state_dict is swapped), so it's a single pass per (round, split).

Self-check per site: the best round among the dumped val AUCs (and its value) must match the round
and val AUC the executor recorded in <site>_best_val_overall_gmic_metrics.json -> prints OK / CHECK!.

Run on the GPU box (needs torch + data + crop caches), from the repo root:
    python dump_ditto_perround_preds.py
Then point the notebook's Ditto entry at tag '<out_tag>' with trajectory=True.
"""
import os, sys, re, json, csv, glob, inspect, logging
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = "/raid/home/lsollis/GMIC/GMIC"
CLIENTS = ["UHCC", "HPU", "RSNA-GCP"]
# job_dir -> output prediction tag (kept distinct from the final-round 'ditto'/'ditto_modulewise')
JOBS = {
    f"{REPO}/gmic_job_ditto_sim": "ditto_perround",
    # f"{REPO}/gmic_job_ditto_mw_sim": "ditto_modulewise_perround",   # uncomment when it finishes
}


class FakeCtx:
    def __init__(self, name):
        self._n = name
    def get_identity_name(self):
        return self._n


def _round_of(path):
    mo = re.search(r"_gmic_model_round_(\d+)\.pth$", os.path.basename(path))
    return int(mo.group(1)) if mo else None


def _breast_val_auc(rdir, site, out_tag, r, breast_aggregate):
    paths = glob.glob(os.path.join(rdir, f"{site}_predictions_{out_tag}_round{r}_val.csv"))
    if not paths:
        return float("nan")
    rows = list(csv.DictReader(open(paths[0], newline="")))
    bp, by = breast_aggregate([x["exam_id"] for x in rows], [x["view"] for x in rows],
                              [float(x["prob_malignant"]) for x in rows], [int(x["label"]) for x in rows])
    return roc_auc_score(by, bp) if len(set(by.tolist())) > 1 else float("nan")


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
        ex.initialize()                            # build data_loader + model ONCE (reuses crop cache)

        rdir = ex.results_dir
        ckpts = sorted(
            [(_round_of(p), p) for p in glob.glob(os.path.join(rdir, f"{site}_gmic_model_round_*.pth"))],
            key=lambda t: (t[0] if t[0] is not None else -1),
        )
        ckpts = [(r, p) for r, p in ckpts if r is not None]
        print(f"  found {len(ckpts)} per-round model checkpoints in {rdir}")
        if not ckpts:
            print("  -> NONE found; the run did not save per-round personal models. Skipping site.")
            del ex; continue

        model = ex._underlying.to(ex.device).eval()
        fctx = FakeCtx(site)
        for r, p in ckpts:
            sd = torch.load(p, map_location=ex.device)
            sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
            model.load_state_dict(sd, strict=False)
            for split in ("val", "test"):
                ex._dump_predictions(fctx, split=split, model=model, round_idx=r,
                                     method_tag=out_tag, save_saliency=False)

        # self-check: my best dumped-val round/AUC vs the executor's recorded best_val_overall
        rounds = [r for r, _ in ckpts]
        aucs = {r: _breast_val_auc(rdir, site, out_tag, r, breast_aggregate) for r in rounds}
        best_r = max(aucs, key=lambda r: (aucs[r] if np.isfinite(aucs[r]) else -1))
        mjson = os.path.join(rdir, f"{site}_best_val_overall_gmic_metrics.json")
        rec = json.load(open(mjson)) if os.path.isfile(mjson) else {}
        rec_auc = rec.get("val_auc", rec.get("auc")); rec_round = rec.get("round")
        ok = (isinstance(rec_auc, (int, float)) and abs(aucs[best_r] - rec_auc) < 1e-2)
        print(f"  dumped {len(rounds)} rounds; my best val: round={best_r} auc={aucs[best_r]:.4f}  "
              f"| recorded best_val_overall: round={rec_round} auc={rec_auc}  "
              f"{'OK' if ok else 'CHECK! (val AUC mismatch -> do not trust)'}")
        del ex, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    for job_dir, out_tag in JOBS.items():
        run_job(job_dir, out_tag)
    print("\nDone. In the notebook, set the Ditto entry to tag '<...>_perround' with trajectory=True.")
