import io
import json
from datetime import datetime, timezone

from PIL import Image, ImageEnhance, ImageFilter

from app.agent.state import VFXJobState
from app.agent.timing import start_timer, elapsed_ms

_NODE = "filter"

# Map action keywords → Pillow operations
_ACTION_MAP = {
    "greyscale":  lambda img: img.convert("L").convert("RGB"),
    "grayscale":  lambda img: img.convert("L").convert("RGB"),
    "sharpen":    lambda img: img.filter(ImageFilter.SHARPEN),
    "blur":       lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
    "contrast":   lambda img: ImageEnhance.Contrast(img).enhance(1.5),
    "brightness": lambda img: ImageEnhance.Brightness(img).enhance(1.3),
    "warm":       lambda img: _tint(img, r=1.15, g=1.05, b=0.90),
    "cool":       lambda img: _tint(img, r=0.90, g=1.05, b=1.15),
}


def _tint(img: Image.Image, r: float, g: float, b: float) -> Image.Image:
    import numpy as np
    arr = np.array(img.convert("RGB")).astype(float)
    arr[..., 0] = np.clip(arr[..., 0] * r, 0, 255)
    arr[..., 1] = np.clip(arr[..., 1] * g, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * b, 0, 255)
    return Image.fromarray(arr.astype("uint8"), "RGB")


def filter_node(state: VFXJobState) -> VFXJobState:
    started_at, t0 = start_timer()

    nodes = state.get("extracted_intent", {}).get("required_nodes", [])
    if _NODE not in nodes:
        return state

    src_bytes = state.get("final_image") or state.get("original_image", b"")

    try:
        img     = Image.open(io.BytesIO(src_bytes)).convert("RGB")
        actions = state.get("extracted_intent", {}).get("actions", [])
        applied = []

        for action in actions:
            fn = _ACTION_MAP.get(action.lower())
            if fn:
                img     = fn(img)
                applied.append(action)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        state["final_image"]  = buf.getvalue()
        state["image_format"] = "png"
        state["status"]       = "done"

    except Exception as exc:
        state["errors"].append({
            "node": _NODE, "error_code": "FILTER_FAILED",
            "message": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        state["status"] = "failed"
        applied = []

    state["nodes_executed"].append(_NODE)
    state.setdefault("node_timings", {})[_NODE] = {
        "started_at": started_at, "duration_ms": elapsed_ms(t0)
    }
    print(json.dumps({"job_id": state["job_id"], "node": _NODE,
                      "event": "complete", "filters_applied": applied,
                      "duration_ms": elapsed_ms(t0), "timestamp": started_at}))
    return state
