import asyncio
import base64
import io
import json
from datetime import datetime, timezone

import httpx
import numpy as np
from PIL import Image

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms
from app.services.replicate_client import run_model, SDXL_CONTROLNET_MODEL

_NODE = "compositing"


def _extract_lighting_hint(image_bytes: bytes) -> str:
    """
    Sample the perimeter pixels of the image to estimate ambient color temperature.
    Perimeter pixels (sky, walls, ground edges) are the dominant source of
    scene-level lighting information. Returns a descriptor injected into the
    SDXL prompt so the generated content respects the original scene's lighting.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    arr = np.array(img, dtype=float)

    # Sample all four edges
    border = np.concatenate([
        arr[0, :],    # top row
        arr[-1, :],   # bottom row
        arr[:, 0],    # left column
        arr[:, -1],   # right column
    ])
    r_mean = border[:, 0].mean()
    g_mean = border[:, 1].mean()
    b_mean = border[:, 2].mean()
    brightness = (r_mean + g_mean + b_mean) / 3.0

    if r_mean > b_mean * 1.3:
        return "warm golden-hour lighting, orange-tinted ambient light, long shadows"
    elif b_mean > r_mean * 1.3:
        return "cool overcast lighting, desaturated blue-grey ambient, diffused shadows"
    elif brightness > 200:
        return "bright midday sunlight, high-key neutral lighting, short hard shadows"
    elif brightness < 80:
        return "low-key dim lighting, dark moody atmosphere, minimal ambient light"
    else:
        return "soft diffused natural lighting, neutral color temperature, gentle shadows"


def _build_prompt(intent: dict, image_bytes: bytes) -> tuple[str, str]:
    objects  = ", ".join(intent.get("target_objects", []))
    actions  = " ".join(intent.get("actions", []))
    lighting = _extract_lighting_hint(image_bytes)
    positive = (
        f"{objects} {actions}, {lighting}, "
        f"photorealistic, cinematic compositing, color-matched, 8k detail"
    )
    negative = "low quality, blurry, distorted, artifacts, flat, cartoon, mismatched lighting"
    return positive.strip(), negative


def _union_mask(masks: dict) -> bytes:
    """Combine all per-object masks into one binary mask (white = replace)."""
    combined: np.ndarray | None = None
    for mask_bytes in masks.values():
        arr = np.array(Image.open(io.BytesIO(mask_bytes)).convert("L"))
        combined = arr if combined is None else np.maximum(combined, arr)
    buf = io.BytesIO()
    Image.fromarray(combined).save(buf, format="PNG")
    return buf.getvalue()


async def _run(state: VFXJobState) -> VFXJobState:
    started_at, t0 = start_timer()

    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if _NODE not in nodes:
        return state

    pos_prompt, neg_prompt = _build_prompt(
        state.get("extracted_intent", {}),
        state["original_image"],
    )
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
        payload["control_image"]                = base64.b64encode(state["depth_map"]).decode()
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
