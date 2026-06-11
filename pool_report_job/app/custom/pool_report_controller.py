"""Server workflow for the pooling job: ONE broadcast round -> collect each site's breast-level
(prob, label) for the shared-global predictions -> concatenate across sites -> compute the POOLED
breast-level AUC (DeLong 95% CI) + Youden(val)->test sens/spec (Wilson 95% CI) via salvage_metrics ->
log one `[salvage-pooled]` line per (method, round) to the SERVER log (readable through the admin
console) and write a durable JSON.

This is the piece the stock FedAvg ModelController could not do via an `aggregator` component (it
ignores that component). A custom ModelController, by contrast, receives every client's FLModel result
directly, so it can pool and report. No persistor/model is needed (persistor_id="" disables it).
"""
import json
import os

import numpy as np

from nvflare.app_common.utils.fl_model_utils import FLModel, ParamsType
from nvflare.app_common.workflows.model_controller import ModelController

from salvage_metrics import endpoint_metrics, format_endpoint


class PoolReportController(ModelController):
    def __init__(self, task_name="pool_report", poolable_methods=None, timeout=900, out_file=None):
        super().__init__(persistor_id="")  # no persistor / model -- this job only gathers + pools
        self.task_name = task_name
        self.poolable = set(poolable_methods) if poolable_methods else {"incoming_global", "pretrained_baseline"}
        self.timeout = int(timeout)
        self.out_file = out_file

    def run(self):
        self.info("[pool-report] broadcasting pool_report task to all clients")
        results = self.send_model_and_wait(
            task_name=self.task_name,
            data=FLModel(params={"_": [0.0]}, params_type=ParamsType.FULL),
            timeout=self.timeout,
        )
        self.info(f"[pool-report] received {len(results)} client result(s)")

        pooled = {}  # "method|round" -> split -> {"p":[], "y":[]}
        for r in results:
            meta = r.meta or {}
            site = meta.get("site", "?")
            raw = meta.get("salvage_preds")
            if not raw:
                self.warning(f"[pool-report] no salvage_preds from {site}")
                continue
            try:
                preds = json.loads(raw) if isinstance(raw, str) else raw
            except Exception as e:
                self.warning(f"[pool-report] bad payload from {site}: {e}")
                continue
            for key, splits in preds.items():
                d = pooled.setdefault(key, {})
                for split, arr in splits.items():
                    dd = d.setdefault(split, {"p": [], "y": []})
                    dd["p"].extend(arr.get("p", []))
                    dd["y"].extend(arr.get("y", []))

        out_rows = []
        for key in sorted(pooled.keys()):
            method, _, rnd = key.partition("|")
            if method not in self.poolable:
                continue
            splits = pooled[key]
            if "val" not in splits or "test" not in splits:
                self.warning(f"[pool-report] {key}: missing val/test across sites; skipping")
                continue
            try:
                m = endpoint_metrics(np.asarray(splits["val"]["p"], float), np.asarray(splits["val"]["y"], int),
                                     np.asarray(splits["test"]["p"], float), np.asarray(splits["test"]["y"], int))
            except Exception as e:
                self.warning(f"[pool-report] pooled compute failed for {key}: {e}")
                continue
            self.info(f"[salvage-pooled] method={method} round={rnd} " + format_endpoint("POOLED", m))
            out_rows.append({"method": method, "round": rnd, **m})

        if not out_rows:
            self.warning("[pool-report] no poolable (method,round) groups produced -- check that all "
                         "sites returned incoming_global predictions")
        if self.out_file and out_rows:
            try:
                os.makedirs(os.path.dirname(self.out_file) or ".", exist_ok=True)
                with open(self.out_file, "w") as f:
                    json.dump(out_rows, f, indent=2)
                self.info(f"[pool-report] wrote {self.out_file}")
            except Exception as e:
                self.warning(f"[pool-report] could not write {self.out_file}: {e}")
        self.info("[pool-report] done")
