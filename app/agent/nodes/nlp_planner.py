import asyncio
import base64
import json
from datetime import datetime, timezone

from app.agent.state import VFXJobState
from app.services.anthropic_client import call_planner

_MEDIA_MAP = {
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
}


async def _run_planner(state: VFXJobState) -> VFXJobState:
    # Build base64 image for vision API
    img_b64    = base64.b64encode(state["original_image"]).decode()
    media_type = _MEDIA_MAP.get(state.get("image_format", "jpeg"), "image/jpeg")

    # Call Claude planner
    plan = await call_planner(img_b64, state["user_prompt"], media_type)

    # Store planner outputs into state
    state["extracted_intent"] = plan
    state["execution_plan"]   = plan.get("plan_summary", "")
    state["plan_confidence"]  = float(plan.get("confidence", 0.0))

    # Clarification guard — do NOT append to nodes_executed if unclear
    if state["plan_confidence"] < 0.7 or plan.get("clarification_needed", False):
        state["status"] = "clarification_required"
        return state

    # Success path
    state["status"] = "running"
    state["nodes_executed"].append("nlp_planner")

    # Emit structured log
    print(json.dumps({
        "job_id":         state["job_id"],
        "node":           "nlp_planner",
        "event":          "complete",
        "confidence":     state["plan_confidence"],
        "required_nodes": plan.get("required_nodes", []),
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }))

    return state


def nlp_planner_node(state: VFXJobState) -> VFXJobState:
    """Sync wrapper — runs the async planner in a dedicated event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run_planner(state))
    finally:
        loop.close()
