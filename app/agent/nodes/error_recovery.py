import json
from datetime import datetime, timezone

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms

_NODE         = "error_recovery"
MAX_RETRIES   = 1

# Nodes that are safe to retry after a transient failure
_RETRYABLE_ERROR_CODES = {
    "SEGMENTATION_FAILED",
    "DEPTH_FAILED",
    "COMPOSITING_FAILED",
    "GRAPH_EXECUTION_ERROR",
}


def error_recovery_node(state: VFXJobState) -> VFXJobState:
    """
    Inspect the most recent error and decide whether to retry.

    If the last error is retryable AND retry_count <= MAX_RETRIES:
      - Increment retry_count
      - Set retry_target to the failing node name
      - Reset status to "running" so graph resumes

    Otherwise: set status="failed" and clear retry_target (terminal).
    """
    started_at, t0 = start_timer()

    errors = state.get("errors", [])
    last   = errors[-1] if errors else {}
    code   = last.get("error_code", "")
    node   = last.get("node", "")

    can_retry = (
        code in _RETRYABLE_ERROR_CODES
        and state.get("retry_count", 0) < MAX_RETRIES
    )

    if can_retry:
        state["retry_count"]  = state.get("retry_count", 0) + 1
        state["retry_target"] = node
        state["status"]       = "running"
    else:
        state["retry_target"] = None
        state["status"]       = "failed"

    state["nodes_executed"].append(_NODE)
    state.setdefault("node_timings", {})[_NODE] = {
        "started_at": started_at, "duration_ms": elapsed_ms(t0)
    }
    print(json.dumps({
        "job_id":       state["job_id"], "node": _NODE,
        "event":        "complete",      "can_retry": can_retry,
        "retry_count":  state.get("retry_count", 0),   # safe — key may be absent in test states
        "retry_target": state.get("retry_target"),
        "timestamp":    datetime.now(timezone.utc).isoformat(),
    }))
    return state
