"""Process-local server bus handing POOLED salvage rows from the aggregator (which computes them in
aggregate()) to the shareable generator (which injects them into the NEXT broadcast so clients get a
copy). Both components run in the same server process, so a module global is the simplest reliable
channel. Server-side only; never imported by clients."""

_PENDING = []


def push(row):
    """Aggregator: queue a pooled row for delivery to clients on the next broadcast."""
    _PENDING.append(row)


def drain():
    """Shareable generator: take everything queued so far (and clear)."""
    global _PENDING
    rows, _PENDING = _PENDING, []
    return rows
