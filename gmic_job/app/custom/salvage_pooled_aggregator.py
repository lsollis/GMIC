"""Server-side aggregator for the FedProx salvage job: standard weighted FedAvg aggregation PLUS a
side-channel that logs ALL salvage paper-stats to the SERVER, whose log persists past job deletion.

Why server-side: the NVFlare job workspace is deleted when the job finishes, and client logs/files
may be unreachable (e.g. the HPU node). The server log is redirected to a mounted path
(NVFL_LOG_ROOT=/workspace/server_logs, see site_folders/server/docker-compose.yml) and tee'd to
server_console.log, so anything logged here survives. Each client ships its de-identified breast-level
(prob, label) for the models it evaluated in the `salvage_stats` shareable header; this aggregator:

  - accept() (per client, per round): logs that client's PER-SITE row for each model
    ([salvage-site] site=... method=... src_round=...), and buffers incoming_global (the shared
    global) for pooling. Per-site deployed_local / incoming_global both land in the server log.
  - aggregate() (round end): logs the POOLED row across sites for the shared global
    ([salvage-pooled] src_round=...), then clears the buffer.

Optionally also appends every row as JSON to {stats_out_dir}/salvage_stats.jsonl (a durable,
machine-readable copy; set stats_out_dir to a mounted path like /workspace/server_logs). The whole
side-channel is defensive: any error logs a warning and the normal weighted aggregation proceeds.

Drop-in superset of InTimeAccumulateWeightedAggregator (same args + optional stats_out_dir).
"""
import json
import os

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import (
    InTimeAccumulateWeightedAggregator,
)

import salvage_bus
from salvage_metrics import endpoint_metrics, format_endpoint

HEADER = "salvage_stats"


class SalvagePooledAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(self, *args, stats_out_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._stats_out_dir = stats_out_dir
        self._pool = {}  # src_round -> {split -> {"p": [...], "y": [...]}}

    # --- helpers ---------------------------------------------------------------------------- #
    def _emit(self, fl_ctx, scope, method, src_round, site, m):
        """Log one stats row to the (persistent) server log and optionally append it as JSON."""
        self.log_info(
            fl_ctx,
            f"[salvage-{scope}] method={method} src_round={src_round} site={site} "
            + format_endpoint(site, m),
        )
        if self._stats_out_dir:
            try:
                os.makedirs(self._stats_out_dir, exist_ok=True)
                rec = {"scope": scope, "method": method, "src_round": int(src_round), "site": site, **m}
                with open(os.path.join(self._stats_out_dir, "salvage_stats.jsonl"), "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                self.log_warning(fl_ctx, f"[salvage] jsonl append failed (ignored): {e}")

    @staticmethod
    def _endpoint_from_splits(splits):
        vp = np.asarray(splits["val"]["p"], dtype=float)
        vy = np.asarray(splits["val"]["y"], dtype=int)
        tp = np.asarray(splits["test"]["p"], dtype=float)
        ty = np.asarray(splits["test"]["y"], dtype=int)
        return endpoint_metrics(vp, vy, tp, ty)

    # --- aggregator overrides --------------------------------------------------------------- #
    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            raw = shareable.get_header(HEADER)
            if raw:
                for rec in (json.loads(raw) if isinstance(raw, str) else raw):
                    method = rec.get("method", "?")
                    sr = int(rec["src_round"])
                    site = rec.get("site", "?")
                    splits = rec.get("splits", {})
                    if "val" in splits and "test" in splits:
                        self._emit(fl_ctx, "site", method, sr, site, self._endpoint_from_splits(splits))
                    # Buffer the SHARED global for pooling (per-site models cannot be pooled).
                    if method == "incoming_global":
                        dst = self._pool.setdefault(sr, {})
                        for split, arr in splits.items():
                            d = dst.setdefault(split, {"p": [], "y": []})
                            d["p"].extend(arr.get("p", []))
                            d["y"].extend(arr.get("y", []))
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage] stats harvest failed (ignored): {e}")
        return super().accept(shareable, fl_ctx)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        try:
            for sr, splits in sorted(self._pool.items()):
                if "val" in splits and "test" in splits:
                    m = self._endpoint_from_splits(splits)
                    self._emit(fl_ctx, "pooled", "incoming_global", sr, "POOLED", m)
                    # Queue for delivery back to the clients on the next broadcast (two-way copy).
                    salvage_bus.push({"method": "incoming_global", "src_round": int(sr), **m})
                else:
                    self.log_warning(fl_ctx, f"[salvage-pooled] src_round={sr}: missing val/test from "
                                             f"some site; skipping pooled (have {sorted(splits)})")
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage] pooled compute failed (ignored): {e}")
        finally:
            self._pool.clear()
        return super().aggregate(fl_ctx)
