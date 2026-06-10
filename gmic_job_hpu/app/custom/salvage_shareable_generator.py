"""Drop-in FullModelShareableGenerator that piggybacks the POOLED salvage rows onto each broadcast.

The aggregator computes pooled metrics in aggregate() and queues them on salvage_bus; the controller
builds every round's broadcast task via this generator's learnable_to_shareable() (scatter_and_gather
line ~241), so draining the bus here and setting a header is the reliable server->client channel
(headers on the aggregator's own output are dropped by shareable_to_learnable). Clients read the
`salvage_pooled_results` header in _salvage_round and log + file the pooled rows. Behavior is otherwise
identical to FullModelShareableGenerator, and fully defensive (any error -> plain broadcast).
"""
import json

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.app_common.shareablegenerators.full_model_shareable_generator import (
    FullModelShareableGenerator,
)

import salvage_bus


class SalvageShareableGenerator(FullModelShareableGenerator):
    def learnable_to_shareable(self, model_learnable, fl_ctx: FLContext) -> Shareable:
        shareable = super().learnable_to_shareable(model_learnable, fl_ctx)
        try:
            rows = salvage_bus.drain()
            if rows:
                shareable.set_header("salvage_pooled_results", json.dumps(rows))
        except Exception:
            pass  # never let the stats side-channel block a broadcast
        return shareable
