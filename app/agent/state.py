from typing import TypedDict, Optional, Literal


class VFXJobState(TypedDict):
    # --- Inputs ---
    job_id:           str
    original_image:   bytes               # Raw image bytes
    image_format:     str                 # "jpeg" | "png" | "webp"
    user_prompt:      str

    # --- Planner Outputs ---
    extracted_intent: dict                # {target_objects, actions, required_nodes}
    execution_plan:   str                 # Human-readable plan string
    plan_confidence:  float               # 0.0–1.0; triggers clarification if < 0.7

    # --- Tool Outputs ---
    masks:            Optional[dict]      # {object_name: alpha_mask_bytes}
    mask_confidence:  Optional[float]     # IoU score from SAM
    depth_map:        Optional[bytes]     # Depth image bytes from MiDaS
    final_image:      Optional[bytes]     # Composited output image

    # --- Execution Metadata ---
    nodes_executed:   list[str]           # Ordered log of completed nodes
    errors:           list[dict]          # [{node, error_code, message, timestamp}]
    quality_flags:    list[str]           # e.g., ["low_confidence_mask"]
    status:           Literal["queued", "planning", "running", "done", "failed"]
