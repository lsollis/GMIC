"""Client task for the pooling job: read THIS site's already-written salvage prediction CSVs,
breast-aggregate them, and return the (prob, label) arrays so the server can pool across sites.

No model, no GPU, no cross-site data access -- the salvage already wrote each site's predictions to
its results_dir; this just reads them and ships the de-identified breast-level scores+labels over the
FL channel. That is what lets the server log a true pooled AUC for a site (e.g. HPU) whose files you
cannot reach directly.
"""
import csv
import glob
import json
import os
from collections import defaultdict

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.utils.fl_model_utils import FLModel, FLModelUtils, ParamsType

from salvage_metrics import breast_aggregate


class PoolReportExecutor(Executor):
    def __init__(self, results_dir="/workspace/gmic_results", task_name="pool_report", methods=None):
        super().__init__()
        self.results_dir = results_dir
        self.task_name = task_name
        # deployed_local is a DIFFERENT model per site and must never be pooled; default to the
        # shared-model methods only. None => ship everything (and let the server filter).
        self.methods = set(methods) if methods else {"incoming_global", "pretrained_baseline"}

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != self.task_name:
            return make_reply(ReturnCode.TASK_UNKNOWN)
        site = fl_ctx.get_identity_name()
        groups = defaultdict(lambda: defaultdict(lambda: {"p": [], "y": []}))  # "method|round" -> split -> arrays
        pattern = os.path.join(self.results_dir, f"{site}_predictions_*.csv")
        n_files = 0
        for path in sorted(glob.glob(pattern)):
            try:
                with open(path, newline="") as f:
                    rows = list(csv.DictReader(f))
            except Exception as e:
                self.log_warning(fl_ctx, f"[pool-report] could not read {path}: {e}")
                continue
            if not rows:
                continue
            method = rows[0].get("method", "?")
            if self.methods and method not in self.methods:
                continue
            rnd, split = rows[0].get("round", "?"), rows[0].get("split", "?")
            try:
                bp, by = breast_aggregate([r["site_id"] for r in rows], [r["exam_id"] for r in rows],
                                          [r["view"] for r in rows], [float(r["prob_malignant"]) for r in rows],
                                          [int(r["label"]) for r in rows])
            except Exception as e:
                self.log_warning(fl_ctx, f"[pool-report] aggregate failed for {path}: {e}")
                continue
            g = groups[f"{method}|{rnd}"][split]
            g["p"].extend(float(x) for x in bp)
            g["y"].extend(int(x) for x in by)
            n_files += 1
        self.log_info(fl_ctx, f"[pool-report] {site}: aggregated {n_files} prediction file(s) -> "
                              f"{len(groups)} (method,round) group(s) [pattern={pattern}]")
        payload = {k: {s: dict(v) for s, v in splits.items()} for k, splits in groups.items()}
        fl_model = FLModel(params={"_": [0.0]}, params_type=ParamsType.FULL,
                           meta={"site": site, "salvage_preds": json.dumps(payload)})
        reply = FLModelUtils.to_shareable(fl_model)
        reply.set_return_code(ReturnCode.OK)
        return reply
