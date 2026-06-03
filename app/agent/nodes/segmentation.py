import asyncio
import base64
import json
from datetime import datetime, timezone

import httpx

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms
from app.services.replicate_client import run_model, GROUNDED_SAM_MODEL

_NODE = "segmentation"


async def _run(state: VFXJobState) -> VFXJobState:
    started_at, t0 = start_timer()

    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if _NODE not in nodes:
        return state

    img_b64 = base64.b64encode(state["original_image"]).decode()
    targets = state["extracted_intent"].get("target_objects", [])

    try:
        # Grounded-SAM accepts free-text labels via text_prompt.
        # It internally uses Grounding DINO for detection, then SAM for masking.
        output = await run_model(
            GROUNDED_SAM_MODEL,
            {
                "image":       img_b64,
                "text_prompt": " . ".join(targets),
            }
        )

        # output[-1] is the combined binary mask URL
        mask_url  = str(output[-1]) if isinstance(output, list) else str(output)
        mask_data = httpx.get(mask_url, timeout=30).content
        masks     = {label: mask_data for label in targets}

        # Confidence proxy: check mask area vs image area
        from PIL import Image
        import io
        first_mask = Image.open(io.BytesIO(mask_data))
        img_area   = first_mask.width * first_mask.height
        mask_area  = sum(1 for p in first_mask.getdata() if p > 127)
        confidence = min(mask_area / max(img_area, 1), 1.0)
        confidence = 0.4 if confidence < 0.01 else confidence

        state["masks"]           = masks
        state["mask_confidence"] = confidence
        if confidence < 0.6:
            state["quality_flags"].append("low_confidence_mask")

    except Exception as exc:
        state["errors"].append({
            "node": _NODE, "error_code": "SEGMENTATION_FAILED",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    state["nodes_executed"].append(_NODE)
    state.setdefault("node_timings", {})[_NODE] = {
        "started_at": started_at, "duration_ms": elapsed_ms(t0)
    }
    print(json.dumps({"job_id": state["job_id"], "node": _NODE,
                      "event": "complete", "duration_ms": elapsed_ms(t0),
                      "timestamp": started_at}))
    return state


def segmentation_node(state: VFXJobState) -> VFXJobState:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(state))
    finally:
        loop.close()
