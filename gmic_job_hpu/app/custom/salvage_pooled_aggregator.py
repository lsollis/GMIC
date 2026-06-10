"""Server-side aggregator for the FedProx salvage job: standard weighted FedAvg aggregation PLUS a
side-channel that computes the POOLED breast-level AUC (DeLong 95% CI) and the Youden-J operating
point (selected on pooled VAL, applied to pooled TEST, sens/spec with Wilson 95% CIs) across all
sites, logged to the SERVER log every round.

Pooled is the one endpoint that needs cross-site fusion -- per-site endpoints are computed and logged
at each client. Each client ships its de-identified breast-level (prob, label) for the reconstructed
shared global in the `salvage_pooled_preds` shareable header; this aggregator harvests them in
accept(), and at aggregate() (round end) computes+logs the pooled metrics for that round's global,
then clears the buffer. The side-channel is fully defensive: any error logs a warning and the normal
weighted aggregation proceeds unchanged.

Drop-in replacement for InTimeAccumulateWeightedAggregator -- same args (expected_data_kind, etc.).
"""
import json

import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.aggregators.intime_accumulate_model_aggregator import (
    InTimeAccumulateWeightedAggregator,
)

from salvage_metrics import endpoint_metrics

HEADER = "salvage_pooled_preds"


class SalvagePooledAggregator(InTimeAccumulateWeightedAggregator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pool = {}  # src_round -> {split -> {"p": [...], "y": [...]}}

    def accept(self, shareable: Shareable, fl_ctx: FLContext) -> bool:
        try:
            raw = shareable.get_header(HEADER)
            if raw:
                rec = json.loads(raw) if isinstance(raw, str) else raw
                sr = int(rec["src_round"])
                dst = self._pool.setdefault(sr, {})
                for split, arr in rec.get("splits", {}).items():
                    d = dst.setdefault(split, {"p": [], "y": []})
                    d["p"].extend(arr.get("p", []))
                    d["y"].extend(arr.get("y", []))
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage-pooled] harvest failed (ignored): {e}")
        return super().accept(shareable, fl_ctx)

    def aggregate(self, fl_ctx: FLContext) -> Shareable:
        try:
            self._log_pooled(fl_ctx)
        except Exception as e:
            self.log_warning(fl_ctx, f"[salvage-pooled] pooled compute failed (ignored): {e}")
        finally:
            self._pool.clear()
        return super().aggregate(fl_ctx)

    def _log_pooled(self, fl_ctx: FLContext):
        for sr, splits in sorted(self._pool.items()):
            if "val" not in splits or "test" not in splits:
                self.log_warning(
                    fl_ctx, f"[salvage-pooled] src_round={sr}: missing val/test from some site; "
                            f"skipping pooled (have splits={sorted(splits)})")
                continue
            vp = np.asarray(splits["val"]["p"], dtype=float)
            vy = np.asarray(splits["val"]["y"], dtype=int)
            tp = np.asarray(splits["test"]["p"], dtype=float)
            ty = np.asarray(splits["test"]["y"], dtype=int)
            m = endpoint_metrics(vp, vy, tp, ty)
            self.log_info(
                fl_ctx,
                f"[salvage-pooled] POOLED src_round={sr} n={m['n_breasts']} (pos={m['n_pos']}) "
                f"valAUC={m['val_auc']:.4f} testAUC={m['test_auc']:.4f} "
                f"[{m['auc_lo']:.4f}-{m['auc_hi']:.4f}] thr={m['youden_thr']:.3f} "
                f"Sens={m['sensitivity']:.3f} [{m['sens_lo']:.3f}-{m['sens_hi']:.3f}] "
                f"Spec={m['specificity']:.3f} [{m['spec_lo']:.3f}-{m['spec_hi']:.3f}]"
            )
