#!/usr/bin/env python3
"""Offline paper-statistics for the FedProx salvage: breast-level AUC with DeLong 95% CIs,
Youden-J operating point (selected on VALIDATION, applied to TEST) with sensitivity/specificity
and their Wilson 95% CIs, computed PER SITE and POOLED.

Input: a directory of per-(site, model, round, split) prediction CSVs as written by the executor's
`_dump_predictions` (columns: site_id, round, method, split, exam_id, view, path, prob_malignant,
label). Gather the CSVs from all three nodes into one directory, then run this script. Nothing here
needs the model, the data, or GPU -- it is pure post-processing on the saved predictions.

Grouping: rows are grouped by (method, round) using the in-file columns, so reconstructed-global
dumps (method=incoming_global) and as-deployed local dumps (method=deployed_local) are handled
uniformly. For each group we report the TEST endpoint per site and pooled; the Youden threshold for
each endpoint is chosen on the SAME endpoint's VALIDATION predictions (per-site val for per-site
test; pooled val for pooled test), never on test (which would be optimistic).

Breast aggregation matches gmic_job_hpu/app/custom/fl_utils.breast_aggregate: a breast is
(site_id, exam_id, laterality); breast prob = mean of its views' malignant probs; breast label =
max over views. site_id is in the key so exam_id collisions across sites cannot merge breasts.

Usage:
    python tools/salvage_stats.py --pred-dir /path/to/gathered_csvs --out-prefix results/fedprox_salvage
    python tools/salvage_stats.py --selftest      # validate DeLong / Wilson / Youden, no inputs needed
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from collections import OrderedDict

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    pd = None
    _PANDAS_ERR = e

try:
    from scipy.stats import norm as _norm
    def _z(alpha):
        return float(_norm.ppf(1.0 - alpha / 2.0))
except Exception:  # scipy optional; default to the 95% z if unavailable
    def _z(alpha):
        if abs(alpha - 0.05) < 1e-9:
            return 1.959963984540054
        # crude fallback for other alphas via inverse-erf-free approximation
        raise RuntimeError("scipy not available; only alpha=0.05 supported without it")


# --------------------------------------------------------------------------- #
# Breast-level aggregation (site-aware; mirrors fl_utils.breast_aggregate)
# --------------------------------------------------------------------------- #
def _parse_laterality(view):
    s = str(view).strip().upper()
    if not s or s[0] not in ("L", "R"):
        raise ValueError(f"cannot parse laterality from view {view!r} (expected L-*/R-*)")
    return s[0]


def breast_aggregate(site_ids, exam_ids, views, probs, labels):
    """View-level rows -> breast-level (prob=mean, label=max), keyed by (site, exam, laterality)."""
    groups = OrderedDict()
    sites = []
    for sid, eid, v, p, y in zip(site_ids, exam_ids, views, probs, labels):
        key = (sid, eid, _parse_laterality(v))
        g = groups.get(key)
        if g is None:
            g = {"p": [], "y": [], "site": sid}
            groups[key] = g
            sites.append(sid)
        g["p"].append(float(p))
        g["y"].append(int(y))
    bp = np.array([np.mean(g["p"]) for g in groups.values()], dtype=float)
    by = np.array([int(max(g["y"])) for g in groups.values()], dtype=int)
    bs = np.array([g["site"] for g in groups.values()], dtype=object)
    return bp, by, bs


# --------------------------------------------------------------------------- #
# DeLong AUC variance (fast algorithm, Sun & Xu 2014) -> AUC + CI
# --------------------------------------------------------------------------- #
def _midrank(x):
    """1-based midranks (ties averaged)."""
    J = np.argsort(x, kind="mergesort")
    Z = x[J]
    N = x.shape[0]
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_auc_ci(labels, scores, alpha=0.05):
    """Single-classifier AUC with a DeLong 95% CI.

    Returns (auc, lo, hi, n_pos, n_neg). Degenerate single-class input -> (nan, nan, nan, ...).
    Logit transform is applied to the CI endpoints so they stay inside [0, 1] (standard for AUC).
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m, n = pos.shape[0], neg.shape[0]
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), m, n

    tx = _midrank(pos)
    ty = _midrank(neg)
    txy = _midrank(np.concatenate([pos, neg]))
    txy_pos, txy_neg = txy[:m], txy[m:]

    auc = (txy_pos.sum() - m * (m + 1) / 2.0) / (m * n)

    v01 = (txy_pos - tx) / n           # structural components over positives (len m)
    v10 = 1.0 - (txy_neg - ty) / m     # over negatives (len n)
    s01 = np.var(v01, ddof=1) if m > 1 else 0.0
    s10 = np.var(v10, ddof=1) if n > 1 else 0.0
    var_auc = s01 / m + s10 / n
    se = math.sqrt(var_auc) if var_auc > 0 else 0.0
    z = _z(alpha)

    if se == 0.0 or auc in (0.0, 1.0):
        return float(auc), float(auc), float(auc), m, n

    # CI on the logit scale, then back-transform (keeps endpoints in (0,1))
    eps = 1e-12
    a = min(max(auc, eps), 1 - eps)
    logit = math.log(a / (1 - a))
    se_logit = se / (a * (1 - a))
    lo = 1.0 / (1.0 + math.exp(-(logit - z * se_logit)))
    hi = 1.0 / (1.0 + math.exp(-(logit + z * se_logit)))
    return float(auc), float(lo), float(hi), m, n


# --------------------------------------------------------------------------- #
# Youden threshold (on val) + sens/spec with Wilson CIs (on test)
# --------------------------------------------------------------------------- #
def youden_threshold(labels, scores):
    """Threshold (>=) maximizing Youden's J = sens + spec - 1 on the given (val) data.

    Candidate cut-points are the distinct scores plus a touch below the minimum (so the all-positive
    operating point is reachable). Ties in J resolved toward the higher threshold (more specific)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    if P == 0 or N == 0:
        return 0.5
    cands = np.unique(scores)
    cands = np.concatenate([[cands[0] - 1e-9], cands])
    best_j, best_t = -np.inf, 0.5
    for t in cands:
        pred = scores >= t
        tp = int(np.sum(pred & (labels == 1)))
        tn = int(np.sum(~pred & (labels == 0)))
        sens = tp / P
        spec = tn / N
        j = sens + spec - 1.0
        if j > best_j or (abs(j - best_j) < 1e-12 and t > best_t):
            best_j, best_t = j, float(t)
    return best_t


def wilson_ci(k, n, alpha=0.05):
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def sens_spec_at(labels, scores, threshold):
    """Sensitivity/specificity (>= threshold) plus the (k, n) behind each, for Wilson CIs."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    pred = scores >= threshold
    P = int((labels == 1).sum())
    N = int((labels == 0).sum())
    tp = int(np.sum(pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    sens = tp / P if P else float("nan")
    spec = tn / N if N else float("nan")
    sens_lo, sens_hi = wilson_ci(tp, P)
    spec_lo, spec_hi = wilson_ci(tn, N)
    return {
        "threshold": float(threshold),
        "sensitivity": sens, "sens_lo": sens_lo, "sens_hi": sens_hi, "tp": tp, "n_pos": P,
        "specificity": spec, "spec_lo": spec_lo, "spec_hi": spec_hi, "tn": tn, "n_neg": N,
    }


# --------------------------------------------------------------------------- #
# Endpoint analysis: Youden on val -> applied to test, with DeLong AUC CI
# --------------------------------------------------------------------------- #
def analyze_endpoint(val_df, test_df, label):
    """One reportable row: AUC+DeLong CI on test, Youden(val)->test sens/spec + Wilson CIs."""
    vp, vy, _ = breast_aggregate(val_df["site_id"], val_df["exam_id"], val_df["view"],
                                 val_df["prob_malignant"], val_df["label"])
    tp_, ty_, _ = breast_aggregate(test_df["site_id"], test_df["exam_id"], test_df["view"],
                                   test_df["prob_malignant"], test_df["label"])
    thr = youden_threshold(vy, vp)
    auc, lo, hi, n_pos, n_neg = delong_auc_ci(ty_, tp_)
    val_auc, _, _, _, _ = delong_auc_ci(vy, vp)   # for pooled-best ROUND selection (select on val)
    op = sens_spec_at(ty_, tp_, thr)
    return {
        "endpoint": label,
        "n_breasts": int(n_pos + n_neg),
        "n_pos": int(n_pos),
        "val_auc": val_auc,
        "test_auc": auc, "auc_lo": lo, "auc_hi": hi,
        "youden_thr": thr,
        "sensitivity": op["sensitivity"], "sens_lo": op["sens_lo"], "sens_hi": op["sens_hi"],
        "specificity": op["specificity"], "spec_lo": op["spec_lo"], "spec_hi": op["spec_hi"],
        "val_n_breasts": int(len(vy)),
    }


def analyze_group(df_group, allow_pooled=True):
    """All endpoints (each site, and pooled when valid) for one (method, round) group.

    allow_pooled MUST be False when each site evaluated a DIFFERENT model (method=deployed_local):
    pooling predictions across three different models is meaningless. It is True for a single shared
    model broadcast to all sites (incoming_global, pretrained_baseline)."""
    val = df_group[df_group["split"] == "val"]
    test = df_group[df_group["split"] == "test"]
    rows = []
    if len(val) and len(test):
        if allow_pooled:
            rows.append(analyze_endpoint(val, test, "POOLED"))
        for site in sorted(set(test["site_id"]) & set(val["site_id"])):
            rows.append(analyze_endpoint(val[val["site_id"] == site],
                                         test[test["site_id"] == site], site))
    return rows


# --------------------------------------------------------------------------- #
# IO / driver
# --------------------------------------------------------------------------- #
_REQUIRED_COLS = ["site_id", "round", "method", "split", "exam_id", "view", "prob_malignant", "label"]


def load_predictions(pred_dir):
    if pd is None:
        raise RuntimeError(f"pandas is required to load CSVs: {_PANDAS_ERR}")
    paths = sorted(glob.glob(os.path.join(pred_dir, "*predictions*.csv")))
    if not paths:
        raise FileNotFoundError(f"no *predictions*.csv found under {pred_dir}")
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        missing = [c for c in _REQUIRED_COLS if c not in d.columns]
        if missing:
            print(f"[warn] skipping {p}: missing columns {missing}", file=sys.stderr)
            continue
        frames.append(d[_REQUIRED_COLS])
    if not frames:
        raise RuntimeError("no usable prediction CSVs (all missing required columns)")
    df = pd.concat(frames, ignore_index=True)
    print(f"[load] {len(paths)} files, {len(df)} view-level rows, "
          f"{df[['method','round']].drop_duplicates().shape[0]} (method,round) groups", file=sys.stderr)
    return df


def _fmt(row):
    return (f"{row['endpoint']:<10} n={row['n_breasts']:>4} (pos={row['n_pos']:>3})  "
            f"valAUC={row['val_auc']:.4f}  "
            f"testAUC={row['test_auc']:.4f} [{row['auc_lo']:.4f}-{row['auc_hi']:.4f}]  "
            f"thr={row['youden_thr']:.3f}  "
            f"Sens={row['sensitivity']:.3f} [{row['sens_lo']:.3f}-{row['sens_hi']:.3f}]  "
            f"Spec={row['specificity']:.3f} [{row['spec_lo']:.3f}-{row['spec_hi']:.3f}]")


def run(pred_dir, out_prefix=None):
    df = load_predictions(pred_dir)
    all_rows = []
    # Pooling is only valid when every site evaluated the SAME model. deployed_local is per-site
    # (a different model per site), so its pooled endpoint would be meaningless -> per-site only.
    pooled_invalid_methods = {"deployed_local"}
    for (method, rnd), g in df.groupby(["method", "round"]):
        print(f"\n=== method={method} round={rnd} ===")
        for row in analyze_group(g, allow_pooled=method not in pooled_invalid_methods):
            row["method"] = method
            row["round"] = int(rnd)
            all_rows.append(row)
            print("  " + _fmt(row))
    if out_prefix and all_rows:
        out_csv = f"{out_prefix}_metrics.csv"
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
        print(f"\n[out] wrote {out_csv}", file=sys.stderr)
        _write_markdown(all_rows, f"{out_prefix}_table.md")
        print(f"[out] wrote {out_prefix}_table.md", file=sys.stderr)
    return all_rows


def _write_markdown(rows, path):
    lines = ["| method | round | endpoint | n (pos) | test AUC [95% CI] | Youden thr | "
             "Sens [95% CI] | Spec [95% CI] |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['round']} | {r['endpoint']} | {r['n_breasts']} ({r['n_pos']}) | "
            f"{r['test_auc']:.4f} [{r['auc_lo']:.4f}–{r['auc_hi']:.4f}] | {r['youden_thr']:.3f} | "
            f"{r['sensitivity']:.3f} [{r['sens_lo']:.3f}–{r['sens_hi']:.3f}] | "
            f"{r['specificity']:.3f} [{r['spec_lo']:.3f}–{r['spec_hi']:.3f}] |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
# Self-test: validate the estimators against independent references
# --------------------------------------------------------------------------- #
def _selftest():
    rng = np.random.RandomState(0)
    ok = True

    # AUC matches the Mann-Whitney/sklearn definition on random data with ties.
    y = rng.randint(0, 2, size=500)
    s = rng.rand(500) + 0.3 * y                 # signal + noise, with float ties unlikely
    s = np.round(s, 2)                          # force ties to exercise midranks
    auc, lo, hi, npos, nneg = delong_auc_ci(y, s)
    # Reference AUC via rank formula (_midrank already returns ranks in original order)
    ranks = _midrank(s)  # 1-based midranks, aligned with s/y
    R_pos = ranks[y == 1].sum()
    auc_ref = (R_pos - npos * (npos + 1) / 2) / (npos * nneg)
    assert abs(auc - auc_ref) < 1e-9, (auc, auc_ref)
    try:
        from sklearn.metrics import roc_auc_score
        assert abs(auc - roc_auc_score(y, s)) < 1e-9, (auc, roc_auc_score(y, s))
        print("[selftest] AUC matches sklearn.roc_auc_score (with ties): OK")
    except ImportError:
        print("[selftest] AUC matches internal rank formula (sklearn absent): OK")
    assert lo < auc < hi, (lo, auc, hi)
    print(f"[selftest] DeLong CI brackets AUC: {auc:.4f} in [{lo:.4f}, {hi:.4f}]: OK")

    # Wilson CI against a hand value: 8/10 -> [0.4901, 0.9433] (known reference, 95%)
    lo_w, hi_w = wilson_ci(8, 10)
    assert abs(lo_w - 0.4901) < 1e-3 and abs(hi_w - 0.9433) < 1e-3, (lo_w, hi_w)
    print(f"[selftest] Wilson CI 8/10 = [{lo_w:.4f}, {hi_w:.4f}] (ref [0.4901, 0.9433]): OK")

    # Youden picks the perfectly separating threshold when one exists.
    yy = np.array([0, 0, 0, 1, 1, 1])
    ss = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = youden_threshold(yy, ss)
    op = sens_spec_at(yy, ss, t)
    assert op["sensitivity"] == 1.0 and op["specificity"] == 1.0, op
    print(f"[selftest] Youden separates clean data at thr={t:.3f} (sens=spec=1.0): OK")

    # Breast aggregation: two views -> one breast (mean prob, max label), site-keyed.
    bp, by, bs = breast_aggregate(
        ["A", "A", "A", "A"], ["e1", "e1", "e1", "e1"],
        ["L-CC", "L-MLO", "R-CC", "R-MLO"], [0.2, 0.4, 0.6, 0.8], [0, 0, 1, 1])
    assert len(bp) == 2 and abs(bp[0] - 0.3) < 1e-9 and abs(bp[1] - 0.7) < 1e-9, (bp, by)
    assert list(by) == [0, 1], by
    print("[selftest] breast aggregation (mean prob / max label / laterality): OK")

    print("\n[selftest] ALL PASSED" if ok else "[selftest] FAILURES")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", help="directory of gathered prediction CSVs")
    ap.add_argument("--out-prefix", default=None, help="write <prefix>_metrics.csv and <prefix>_table.md")
    ap.add_argument("--selftest", action="store_true", help="run estimator self-tests and exit")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if _selftest() else 1)
    if not args.pred_dir:
        ap.error("--pred-dir is required (or use --selftest)")
    run(args.pred_dir, args.out_prefix)


if __name__ == "__main__":
    main()
