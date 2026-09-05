"""
LOCAL TRACE LOGGER — substitutes for the W&B Weave cloud dashboard.

This sandbox has no network access to api.wandb.ai (verified: weave.init()
raises TransportServerError / 403 Forbidden). The real `weave` package IS
installed and its `@weave.op()` decorators ARE used throughout graph.py —
weave.op works standalone (confirmed) and simply skips remote logging with a
warning when weave.init() hasn't succeeded. This module captures the same
per-call trace data (inputs, outputs, latency, call id) to a local JSONL file
so the eval report and dashboard have real trace data to work from, in place
of the Weave UI.
"""
import json
import time
import uuid
import os

TRACE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trace_log.jsonl")


class TraceLogger:
    def __init__(self, path: str = TRACE_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, run_id: str, node_name: str, inputs: dict, outputs: dict, latency_ms: float):
        record = {
            "trace_id": str(uuid.uuid4()),
            "run_id": run_id,
            "node": node_name,
            "inputs": inputs,
            "outputs": outputs,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return record


_logger = TraceLogger()


def traced(node_name: str):
    """Decorator: times a node function and writes a local trace record,
    keyed by run_id found in the node's state dict."""

    def decorator(fn):
        def wrapper(state: dict):
            start = time.perf_counter()
            result_state = fn(state)
            latency_ms = (time.perf_counter() - start) * 1000
            run_id = state.get("run_id", "unknown")
            _logger.log(
                run_id=run_id,
                node_name=node_name,
                inputs={k: v for k, v in state.items() if k != "run_id"},
                outputs={k: v for k, v in result_state.items() if k not in state or result_state[k] != state.get(k)},
                latency_ms=latency_ms,
            )
            return result_state

        return wrapper

    return decorator
