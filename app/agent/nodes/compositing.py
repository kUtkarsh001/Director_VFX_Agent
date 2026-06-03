import asyncio
import base64
import io
import json
from datetime import datetime, timezone

import httpx
from PIL import Image

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms
from app.services.replicate_client import run_model, SDXL_CONTROLNET_MODEL

_NODE = "compositing"


def _build_prompt(intent: dict) -> tuple[str, str]:
    objects = ", ".join(intent.get("target_objects", []))
    actions = " ".join(intent.get("actions", []))
    positive = f"{objects} {actions}, photorealistic, cinematic lighting, 8k detail"
    negative = "low quality, blurry, distorted, artifacts, flat, cartoon"
    return positive.strip(), negative


def _union_mask(masks: dict) -> bytes:
    """Combine all per-object masks into one binary mask (white = replace)."""
    combined = None
    for mask_bytes in masks.values():
        img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        combined = img if combined is None else Image.fromarray(
            __import__("numpy").maximum(
                __import__("numpy").array(combined),
                __import__("numpy").array(img)
            )
        )
    buf = io.BytesIO()
    combined.save(buf, format="PNG")
    return buf.getvalue()


async def _run(state: VFXJobState) -> VFXJobState:
    started_at, t0 = start_timer()

    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if _NODE not in nodes:
        return state

    pos_prompt, neg_prompt = _build_prompt(state.get("extracted_intent", {}))
    img_b64  = base64.b64encode(state["original_image"]).decode()
    mask_b64 = base64.b64encode(
        _union_mask(state["masks"]) if state.get("masks")
        else _make_full_mask(state["original_image"])
    ).decode()

    payload: dict = {
        "image":           img_b64,
        "mask":            mask_b64,
        "prompt":          pos_prompt,
        "negative_prompt": neg_prompt,
        "num_inference_steps": 30,
        "guidance_scale": 7.5,
    }
    if state.get("depth_map"):
        payload["control_image"]               = base64.b64encode(state["depth_map"]).decode()
        payload["controlnet_conditioning_scale"] = 0.8

    try:
        output   = await run_model(SDXL_CONTROLNET_MODEL, payload)
        out_url  = str(output[0]) if isinstance(output, list) else str(output)
        state["final_image"]  = httpx.get(out_url, timeout=60).content
        state["image_format"] = "png"
        state["status"]       = "done"
    except Exception as exc:
        state["errors"].append({
            "node": _NODE, "error_code": "COMPOSITING_FAILED",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        state["status"] = "failed"

    state["nodes_executed"].append(_NODE)
    state.setdefault("node_timings", {})[_NODE] = {
        "started_at": started_at, "duration_ms": elapsed_ms(t0)
    }
    print(json.dumps({"job_id": state["job_id"], "node": _NODE,
                      "event": "complete", "duration_ms": elapsed_ms(t0),
                      "depth_conditioned": bool(state.get("depth_map")),
                      "timestamp": started_at}))
    return state


def _make_full_mask(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = Image.new("L", img.size, 255)
    buf  = io.BytesIO()
    mask.save(buf, format="PNG")
    return buf.getvalue()


def compositing_node(state: VFXJobState) -> VFXJobState:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_run(state))
    finally:
        loop.close()
