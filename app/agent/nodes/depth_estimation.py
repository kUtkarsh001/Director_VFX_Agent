import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms
from app.services.replicate_client import run_model, MIDAS_MODEL

_NODE = "depth_estimation"


async def _run(state: VFXJobState) -> VFXJobState:
    started_at, t0 = start_timer()

    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if _NODE not in nodes:
        return state

    img_b64 = base64.b64encode(state["original_image"]).decode()

    try:
        output = await run_model(MIDAS_MODEL, {"image": img_b64})

        # MiDaS returns a URL pointing to the 16-bit grayscale depth map PNG
        depth_url = str(output) if not isinstance(output, list) else str(output[0])
        state["depth_map"] = httpx.get(depth_url, timeout=30).content

    except Exception as exc:
        # Depth failure is NON-FATAL — compositing falls back to standard inpainting
        state["errors"].append({
            "node": _NODE, "error_code": "DEPTH_FAILED",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        state["quality_flags"].append("depth_estimation_failed")

    state["nodes_executed"].append(_NODE)
    state.setdefault("node_timings", {})[_NODE] = {
        "started_at": started_at, "duration_ms": elapsed_ms(t0)
    }
    print(json.dumps({"job_id": state["job_id"], "node": _NODE,
                      "event": "complete", "duration_ms": elapsed_ms(t0),
                      "timestamp": started_at}))
    return state


def depth_estimation_node(state: VFXJobState) -> VFXJobState:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(state))
    finally:
        loop.close()
