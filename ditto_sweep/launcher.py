#!/usr/bin/env python3
"""Ditto lambda sweep via the NVFLARE simulator on one multi-GPU host (e.g. the DGX).

For each ditto_lambda it renders a job from a base job (custom code) + the ditto config
templates, runs `nvflare simulator` with 3 simulated clients (UHCC/HPU/RSNA-GCP) each
pinned to its own GPU, collects each client's best PERSONAL-model val AUC, and reports the
lambda that maximizes the WORST-site AUC (the equity-thesis selection metric). All three
aggregates (worst_site / mean_site / pooled-as-mean) are logged.

Key design
----------
- Per-client isolation on the shared sim filesystem comes from the executor's
  data_path_map (CSVs) + preprocess_cache_dir_map (crops) + {site} path substitution.
- The crop cache is FIXED per site and SHARED across all lambda runs, so preprocessing
  happens once. A warmup run (num_rounds=1) builds the caches before any parallel runs to
  avoid a cold-cache write race; auto-skipped if the caches already exist.
- Up to len(gpu_pools) lambdas run concurrently (default 2), each pool driving one
  simulator process across 3 GPUs.

This script can't be exercised on the Windows clone (no GPUs / real data); run it on the
DGX. It only shells out to `nvflare simulator` and reads JSON the executor writes.

Example
-------
  python launcher.py \
    --base-job ../gmic_job \
    --lambdas 0.01,0.05,0.1,0.5,1.0,2.0 \
    --rounds 20 \
    --gpu-pools "1,2,3;4,5,6" \
    --output-base /workspace/sim/ditto \
    --work-root  /workspace/sim/ditto_runs
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time


# --------------------------------------------------------------------------- IO helpers
def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def _lam_str(lam: float) -> str:
    """Stable, filename-safe label for a lambda value (0.1 -> '0.1', 1.0 -> '1')."""
    s = ("%g" % lam)
    return s


# --------------------------------------------------------------------------- job rendering
def render_job(lam, rounds, base_job, templates_dir, work_root, output_base, clients):
    """Materialize a runnable simulator job dir for one lambda.

    Copies base_job/app/custom (the shared model/executor code) and writes the three
    rendered config files. Per-lambda dynamic fields (ditto_lambda + output/tb/log paths)
    are set here; {site} is left intact for the executor to substitute per client.
    Returns the job dir path.
    """
    ls = _lam_str(lam)
    job_dir = os.path.join(work_root, f"job_l{ls}")
    app_dir = os.path.join(job_dir, "app")
    cfg_dir = os.path.join(app_dir, "config")
    os.makedirs(cfg_dir, exist_ok=True)

    # 1) custom code (executor, model, data_loader, ...) from the base job, verbatim
    src_custom = os.path.join(base_job, "app", "custom")
    dst_custom = os.path.join(app_dir, "custom")
    if os.path.isdir(dst_custom):
        shutil.rmtree(dst_custom)
    shutil.copytree(src_custom, dst_custom)

    # 2) client config: set ditto_lambda + per-lambda output paths ({site} preserved)
    client_cfg = _load_json(os.path.join(templates_dir, "config_fed_client.json"))
    args = client_cfg["executors"][0]["executor"]["args"]
    args["method"] = "ditto"
    args["ditto_lambda"] = float(lam)
    run_root = f"{output_base}/l{ls}"
    args["output_dir"] = f"{run_root}/{{site}}"
    args["tb_log_dir"] = f"{run_root}/tb/{{site}}"
    args["log_file"] = f"{run_root}/{{site}}/executor.log"
    _dump_json(os.path.join(cfg_dir, "config_fed_client.json"), client_cfg)

    # 3) server config: set num_rounds + num_clients
    server_cfg = _load_json(os.path.join(templates_dir, "config_fed_server.json"))
    wf_args = server_cfg["workflows"][0]["args"]
    wf_args["num_rounds"] = int(rounds)
    wf_args["num_clients"] = len(clients)
    _dump_json(os.path.join(cfg_dir, "config_fed_server.json"), server_cfg)

    # 4) meta.json
    meta = _load_json(os.path.join(templates_dir, "meta.json"))
    meta["min_clients"] = len(clients)
    meta["mandatory_clients"] = list(clients)
    _dump_json(os.path.join(job_dir, "meta.json"), meta)

    return job_dir


def cache_paths(templates_dir, clients):
    """Per-site preprocess cache dirs from the template's preprocess_cache_dir_map."""
    args = _load_json(os.path.join(templates_dir, "config_fed_client.json"))[
        "executors"][0]["executor"]["args"]
    cmap = args.get("preprocess_cache_dir_map", {})
    return {c: cmap.get(c) for c in clients}


def caches_ready(templates_dir, clients):
    """True iff every site already has a built crop cache (processed_exam_list.pkl)."""
    for c, d in cache_paths(templates_dir, clients).items():
        if not d or not os.path.isfile(os.path.join(d, "processed_exam_list.pkl")):
            return False
    return True


# --------------------------------------------------------------------------- running
def simulator_cmd(job_dir, workspace, clients, gpus, threads):
    return [
        "nvflare", "simulator", job_dir,
        "-w", workspace,
        "-n", str(len(clients)),
        "-t", str(threads),
        "-c", ",".join(clients),
        "--gpu", ",".join(str(g) for g in gpus),
    ]


def launch(job_dir, workspace, clients, gpus, threads, log_path):
    """Start one simulator subprocess; returns (Popen, log_file_handle)."""
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lf = open(log_path, "w")
    cmd = simulator_cmd(job_dir, workspace, clients, gpus, threads)
    lf.write("CMD: " + " ".join(cmd) + "\n")
    lf.flush()
    proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
    return proc, lf


# --------------------------------------------------------------------------- metrics
def collect_lambda_metrics(output_base, lam, clients):
    """Read each client's best personal-model val AUC for one lambda run.

    Looks for {output_base}/l{ls}/{site}/{site}_best_val_overall_gmic_metrics.json, falling
    back to {site}_final_results.json. Returns {site: auc_or_None}.
    """
    ls = _lam_str(lam)
    out = {}
    for c in clients:
        site_dir = os.path.join(output_base, f"l{ls}", c)
        auc = None
        primary = os.path.join(site_dir, f"{c}_best_val_overall_gmic_metrics.json")
        if os.path.isfile(primary):
            try:
                m = _load_json(primary)
                auc = m.get("val_auc", m.get("auc"))
            except Exception:
                auc = None
        if auc is None:
            fr = os.path.join(site_dir, f"{c}_final_results.json")
            if os.path.isfile(fr):
                try:
                    m = _load_json(fr)
                    auc = (m.get("best_val_auc")
                           or (m.get("best_metrics") or {}).get("val_auc"))
                except Exception:
                    auc = None
        out[c] = float(auc) if isinstance(auc, (int, float)) else None
    return out


def aggregate(per_site):
    """worst_site / mean_site over the sites that produced a metric."""
    vals = [v for v in per_site.values() if isinstance(v, (int, float))]
    if not vals:
        return {"worst_site": None, "mean_site": None, "n": 0}
    return {"worst_site": min(vals), "mean_site": sum(vals) / len(vals), "n": len(vals)}


# --------------------------------------------------------------------------- reporting
def print_report(results, clients, metric):
    sites_hdr = "  ".join(f"{c:>10}" for c in clients)
    print("\n" + "=" * (34 + 12 * len(clients)))
    print("DITTO LAMBDA SWEEP RESULTS")
    print("=" * (34 + 12 * len(clients)))
    print(f"{'lambda':>8}  {'worst_site':>10}  {'mean_site':>10}   {sites_hdr}")
    ranked = sorted(
        results,
        key=lambda r: (r["agg"].get(metric) if r["agg"].get(metric) is not None else -1),
        reverse=True,
    )
    for r in results:
        agg = r["agg"]
        ws = "%.4f" % agg["worst_site"] if agg["worst_site"] is not None else "   NA"
        ms = "%.4f" % agg["mean_site"] if agg["mean_site"] is not None else "   NA"
        per = "  ".join(
            ("%10.4f" % r["per_site"][c]) if isinstance(r["per_site"].get(c), (int, float))
            else f"{'NA':>10}" for c in clients
        )
        star = "  <- BEST" if (ranked and r is ranked[0] and agg.get(metric) is not None) else ""
        print(f"{_lam_str(r['lambda']):>8}  {ws:>10}  {ms:>10}   {per}{star}")
    if ranked and ranked[0]["agg"].get(metric) is not None:
        best = ranked[0]
        print(f"\nBEST ({metric}): lambda={_lam_str(best['lambda'])}  "
              f"{metric}={best['agg'][metric]:.4f}")
    else:
        print("\nNo lambda produced a usable metric (check run logs).")
    print("=" * (34 + 12 * len(clients)) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Ditto lambda sweep via NVFLARE simulator.")
    ap.add_argument("--base-job", required=True, help="job dir providing app/custom (e.g. ../gmic_job)")
    ap.add_argument("--templates-dir", default=os.path.join(os.path.dirname(__file__), "templates"))
    ap.add_argument("--lambdas", default="0.01,0.05,0.1,0.5,1.0,2.0")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--gpu-pools", default="1,2,3;4,5,6",
                    help="';'-separated GPU triples; one concurrent run per pool")
    ap.add_argument("--clients", default="UHCC,HPU,RSNA-GCP")
    ap.add_argument("--threads", type=int, default=3, help="simulator -t (clients run concurrently)")
    ap.add_argument("--output-base", default="/workspace/sim/ditto",
                    help="output_dir base; executor appends /l<lam>/<site>")
    ap.add_argument("--work-root", default="/workspace/sim/ditto_runs",
                    help="where rendered job dirs + simulator workspaces + run logs go")
    ap.add_argument("--metric", default="worst_site", choices=["worst_site", "mean_site"])
    ap.add_argument("--skip-warmup", action="store_true", help="assume caches are already built")
    ap.add_argument("--poll", type=int, default=20, help="seconds between scheduler polls")
    ap.add_argument("--dry-run", action="store_true", help="render jobs + print plan, don't run")
    a = ap.parse_args()

    clients = [c.strip() for c in a.clients.split(",") if c.strip()]
    lambdas = [float(x) for x in a.lambdas.split(",") if x.strip()]
    pools = [[g.strip() for g in pool.split(",") if g.strip()]
             for pool in a.gpu_pools.split(";") if pool.strip()]
    os.makedirs(a.work_root, exist_ok=True)

    print(f"[sweep] lambdas={lambdas} rounds={a.rounds} clients={clients}")
    print(f"[sweep] gpu_pools={pools} (=> {len(pools)} concurrent runs)  metric={a.metric}")
    print(f"[sweep] output_base={a.output_base}  work_root={a.work_root}")

    # Pre-render all jobs (cheap; also validates templates early)
    jobs = {}
    for lam in lambdas:
        jobs[lam] = render_job(lam, a.rounds, a.base_job, a.templates_dir,
                               a.work_root, a.output_base, clients)
        print(f"[render] lambda={_lam_str(lam)} -> {jobs[lam]}")

    if a.dry_run:
        print("[dry-run] jobs rendered; not launching.")
        for lam in lambdas:
            ws = os.path.join(a.work_root, f"ws_l{_lam_str(lam)}")
            print("  would run:", " ".join(simulator_cmd(jobs[lam], ws, clients, pools[0], a.threads)))
        return

    # Warmup: build the per-site crop caches once (solo) before any parallel runs, unless
    # already present. This prevents two cold parallel runs from racing on the same cache.
    if not a.skip_warmup and not caches_ready(a.templates_dir, clients):
        print("[warmup] crop caches missing -> building once via a 1-round solo run "
              f"(lambda={_lam_str(lambdas[0])}, gpus={pools[0]})")
        warm_job = render_job(lambdas[0], 1, a.base_job, a.templates_dir,
                              a.work_root, a.output_base + "_warmup", clients)
        ws = os.path.join(a.work_root, "ws_warmup")
        proc, lf = launch(warm_job, ws, clients, pools[0], a.threads,
                          os.path.join(a.work_root, "warmup.log"))
        rc = proc.wait()
        lf.close()
        print(f"[warmup] done rc={rc}; caches_ready={caches_ready(a.templates_dir, clients)}")
        if not caches_ready(a.templates_dir, clients):
            print("[warmup] WARNING: caches still not detected; check warmup.log. Continuing.")
    else:
        print("[warmup] skipped (caches present or --skip-warmup).")

    # Schedule the sweep across GPU pools (one running job per pool).
    pending = list(lambdas)
    running = {}  # pool_idx -> (lam, proc, lf, start_ts)
    print(f"[run] starting sweep of {len(pending)} lambdas across {len(pools)} pools")
    while pending or running:
        # fill free pools
        for pi in range(len(pools)):
            if pi not in running and pending:
                lam = pending.pop(0)
                ws = os.path.join(a.work_root, f"ws_l{_lam_str(lam)}")
                log_path = os.path.join(a.work_root, f"run_l{_lam_str(lam)}.log")
                proc, lf = launch(jobs[lam], ws, clients, pools[pi], a.threads, log_path)
                running[pi] = (lam, proc, lf, time.time())
                print(f"[run] launched lambda={_lam_str(lam)} on pool{pi}={pools[pi]} -> {log_path}")
        # reap finished
        for pi in list(running.keys()):
            lam, proc, lf, t0 = running[pi]
            rc = proc.poll()
            if rc is not None:
                lf.close()
                dt = time.time() - t0
                print(f"[run] lambda={_lam_str(lam)} on pool{pi} finished rc={rc} in {dt/60:.1f} min")
                del running[pi]
        if running:
            time.sleep(a.poll)

    # Collect + report
    results = []
    for lam in lambdas:
        per_site = collect_lambda_metrics(a.output_base, lam, clients)
        results.append({"lambda": lam, "per_site": per_site, "agg": aggregate(per_site)})
    print_report(results, clients, a.metric)

    summary_path = os.path.join(a.work_root, "sweep_summary.json")
    _dump_json(summary_path, {
        "metric": a.metric, "rounds": a.rounds, "clients": clients,
        "results": results,
    })
    print(f"[sweep] summary written -> {summary_path}")


if __name__ == "__main__":
    sys.exit(main())
