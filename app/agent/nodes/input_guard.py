import io
from datetime import datetime, timezone
from PIL import Image

from app.agent.state import VFXJobState

SUPPORTED_FORMATS = {"jpeg", "png", "webp"}
MAX_IMAGE_BYTES   = 10 * 1024 * 1024   # 10 MB
MAX_PROMPT_CHARS  = 500
MAX_DIMENSION     = 2048


def _fail(state: VFXJobState, code: str, message: str) -> VFXJobState:
    state["errors"].append({
        "node":       "input_guard",
        "error_code": code,
        "message":    message,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
    })
    state["status"] = "failed"
    return state


def input_guard_node(state: VFXJobState) -> VFXJobState:
    # --- Validate prompt -----------------------------------------------
    prompt = state.get("user_prompt", "")
    if not prompt or not prompt.strip():
        return _fail(state, "PROMPT_INVALID", "Prompt must not be empty.")
    if len(prompt) > MAX_PROMPT_CHARS:
        return _fail(
            state, "PROMPT_INVALID",
            f"Prompt exceeds {MAX_PROMPT_CHARS} characters. "
            f"Received: {len(prompt)}.",
        )

    # --- Validate raw image size ----------------------------------------
    image_bytes = state.get("original_image", b"")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        mb = len(image_bytes) / 1024 / 1024
        return _fail(
            state, "INPUT_TOO_LARGE",
            f"Image file exceeds 10 MB limit. Received: {mb:.1f} MB.",
        )

    # --- Open image and validate format ---------------------------------
    try:
        img = Image.open(io.BytesIO(image_bytes))
        fmt = (img.format or "").lower()
        if fmt == "jpg":
            fmt = "jpeg"
    except Exception as exc:
        return _fail(state, "INPUT_INVALID_FORMAT",
                     f"Cannot open image: {exc}")

    if fmt not in SUPPORTED_FORMATS:
        return _fail(
            state, "INPUT_INVALID_FORMAT",
            f"Unsupported format '{fmt}'. Accepted: jpeg, png, webp.",
        )

    # --- Resize if either dimension exceeds 2048px ----------------------
    if img.width > MAX_DIMENSION or img.height > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=img.format or fmt.upper())
        state["original_image"] = buf.getvalue()

    # --- Success --------------------------------------------------------
    state["image_format"] = fmt
    state["status"]        = "planning"
    state["nodes_executed"].append("input_guard")
    return state
