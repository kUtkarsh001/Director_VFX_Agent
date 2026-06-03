# Architecture Document
## EskillVeda VFX Director Agent
**Version:** 1.1 | **Author:** AI Engineer Candidate | **Date:** 2026-06-03

---

## 1. Architecture Philosophy

The system is built on a single guiding principle: **separate reasoning from execution.** The LLM does not call models directly — it produces a structured plan, and the graph executes it. This separation means the LLM is replaceable, each tool is independently upgradeable, and failures are localised.

The architecture is intentionally conservative in complexity. Every component earns its place by improving reliability, maintainability, or user value. Nothing is added because it is technically interesting.

---

## 2. System Context

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Client Layer                                 │
│   curl / Python SDK / Postman                                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTP
┌────────────────────────────▼────────────────────────────────────────┐
│                        FastAPI Server                               │
│   POST /jobs  │  GET /jobs/{id}  │  GET /jobs/{id}/plan             │
│   GET /jobs/{id}/result  │  GET /jobs/{id}/trace                    │
└────────────────────────────┬────────────────────────────────────────┘
                             │ Python function call
┌────────────────────────────▼────────────────────────────────────────┐
│                   LangGraph Agent (Director)                        │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  Input   │   │  NLP     │   │ Tool     │   │  Error           │ │
│  │  Guard   ├──►│  Planner ├──►│ Nodes    │   │  Recovery        │ │
│  │  Node    │   │  Node    │   │ (2–4)    │   │  Node            │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
                             │ HTTPS API calls
     ┌───────────────────────┼─────────────────────────┐
     │                       │                         │
┌────▼────┐            ┌─────▼──────┐          ┌───────▼──────┐
│ Claude  │            │ Replicate  │          │  HuggingFace │
│ API     │            │ (SAM,      │          │  Inference   │
│ (LLM)   │            │  MiDaS,    │          │  (fallback)  │
└─────────┘            │  SD+CtrlNt)│          └──────────────┘
                        └────────────┘
```

---

## 3. LangGraph State Schema

The state is a typed Python dictionary (`TypedDict`) that flows through every node. Nodes read from and write to this state; they never communicate with each other directly.

```python
from typing import TypedDict, Optional, Literal
from dataclasses import dataclass, field

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
```

**Design rationale:** Keeping all state in one typed object means any node can inspect prior results without coupling to another node. The `quality_flags` list is a first-class citizen — it allows the system to deliver a result *with a caveat* rather than forcing a binary success/failure.

---

## 4. Node Specifications

### 4.1 Node 0 — Input Guard

**Purpose:** Validate and pre-process inputs before they touch the LLM or any paid API. Catching bad inputs here costs nothing.

**Actions:**
1. Validate image format and size (≤ 10 MB, supported MIME type).
2. Validate prompt length (≤ 500 chars) and screen for empty/nonsense input.
3. Resize image if dimensions exceed 2048px on either axis (preserving aspect ratio) to control downstream API cost and latency.
4. Encode image to base64 and store in state.

**Exits:**
- Valid → NLP Planner Node
- Invalid → Terminal failure with structured error (no LLM call made)

---

### 4.2 Node 1 — NLP Planner (The Director)

**Purpose:** The cognitive core of the agent. Transforms free-text intent into a structured execution plan.

**LLM:** Claude (claude-sonnet-4-20250514). Temperature: 0.1 (deterministic, not creative).

**System Prompt Strategy:**
The planner is given a strict output schema and instructed to return a JSON object. The schema is:

```json
{
  "target_objects":  ["person", "sky"],
  "actions":         ["remove_object", "replace_background"],
  "required_nodes":  ["segmentation", "depth_estimation", "compositing"],
  "plan_summary":    "Isolate the person and sky using SAM. Estimate scene depth with MiDaS. Replace the sky with a sci-fi nebula via Stable Diffusion with depth conditioning.",
  "confidence":      0.92,
  "clarification_needed": false,
  "clarification_question": null
}
```

**Clarification Guard:** If `confidence < 0.7` or `clarification_needed: true`, the node returns a `CLARIFICATION_REQUIRED` status and the `clarification_question` string to the API caller without proceeding. This prevents the system from confidently doing the wrong thing.

**Exits:**
- Plan confident → Tool routing (conditional edges)
- Clarification needed → API returns clarification request to user

---

### 4.3 Node 2 — Segmentation (Object Masking)

**Purpose:** Generate precise per-object alpha masks.

**Model:** Segment Anything Model (SAM) via Replicate API.
**Trigger condition:** `"segmentation"` ∈ `state.extracted_intent.required_nodes`

**Process:**
1. Submit the base64-encoded image and `target_objects` list to SAM.
2. Receive binary masks for each named object.
3. Compute mask confidence (IoU estimate from SAM's logits).
4. If `mask_confidence < 0.6`, append `"low_confidence_mask"` to `state.quality_flags` and continue (do not abort).
5. Store masks in `state.masks`.

**Why not abort on low confidence?** Aborting on a marginally uncertain mask is worse UX than delivering a result with a flag. The user can decide if the quality is acceptable.

---

### 4.4 Node 3 — Depth Estimation

**Purpose:** Extract scene geometry to enable physically plausible compositing.

**Model:** MiDaS (via HuggingFace Inference API or Replicate).
**Trigger condition:** `"depth_estimation"` ∈ `state.extracted_intent.required_nodes`

**Process:**
1. Submit the original image to the depth estimation API.
2. Receive a depth map (16-bit greyscale PNG).
3. Store in `state.depth_map`.

**Architectural note:** Depth estimation is intentionally kept as a separate node rather than being absorbed into the compositing node. This preserves modularity — if a better model (e.g., Depth Pro by Apple) releases tomorrow, only this node's API call changes.

---

### 4.5 Node 4 — Compositing & Lighting Matching

**Purpose:** Produce the final VFX output by merging all computed artefacts.

**Model:** Stable Diffusion XL + ControlNet (depth-conditioned) via Replicate.
**Trigger condition:** `"compositing"` ∈ `state.extracted_intent.required_nodes`

**Process:**
1. Assemble the inpainting request:
   - **Image:** Original image
   - **Mask:** Union of relevant object masks from `state.masks`
   - **Depth conditioning:** `state.depth_map` fed to ControlNet
   - **Prompt:** The generative instruction extracted from user intent (e.g., `"sci-fi nebula sky, dramatic lighting"`)
2. Submit to Replicate's ControlNet endpoint.
3. Store output in `state.final_image`.

**Why ControlNet matters:** Without depth conditioning, Stable Diffusion generates a replacement background that ignores the scene's geometry and light direction, producing a "flat paste" artefact. ControlNet forces the model to respect perspective and depth — this is the primary technical differentiator from naive inpainting.

---

### 4.6 Node 5 — Simple Filter Path (Shortcut)

**Purpose:** Handle simple transformations (greyscale, contrast, colour grading) without any external API call.

**Trigger condition:** `required_nodes == ["filter"]`
**Library:** Pillow (no external API cost, < 1s execution)

This node exists for correctness and cost efficiency. A prompt like `"make it black and white"` should never route through three paid API calls.

---

### 4.7 Error Recovery Node

**Purpose:** Centralise failure handling. Triggered by any node that raises an exception.

**Process:**
1. Log the error to `state.errors`.
2. Determine if the failed node is retryable (network timeouts → yes; invalid inputs → no).
3. If retryable and retry count < 1: re-route to the failed node.
4. If not retryable or retry exhausted: set `state.status = "failed"` and route to terminal.

**Design rationale:** Without a dedicated error node, error handling logic would be duplicated across Nodes 2, 3, and 4, creating maintenance debt. Centralising it here makes recovery logic auditable in one place.

---

## 5. Conditional Routing Logic

LangGraph conditional edges replace `if/else` chains in a linear script. The Planner's output drives all routing decisions.

```python
def route_after_planner(state: VFXJobState) -> str:
    nodes = state["extracted_intent"]["required_nodes"]
    if state["plan_confidence"] < 0.7:
        return "clarification_required"      # Terminal — return to API
    if "segmentation" in nodes:
        return "segmentation"
    if "depth_estimation" in nodes:
        return "depth_estimation"
    if "compositing" in nodes:
        return "compositing"
    if "filter" in nodes:
        return "filter"
    return "error_recovery"                  # No valid route found

def route_after_segmentation(state: VFXJobState) -> str:
    if state["errors"]:
        return "error_recovery"
    nodes = state["extracted_intent"]["required_nodes"]
    if "depth_estimation" in nodes:
        return "depth_estimation"
    if "compositing" in nodes:
        return "compositing"
    return "output"

# Similar routing functions for depth and compositing nodes...
```

**Why this matters to evaluators:** The agent is not a linear pipeline that always runs all four steps. It is a decision-making system. A prompt requiring only depth estimation skips segmentation entirely. This is the defining characteristic of an *agent* vs a *script*.

---

## 6. Graph Construction

```python
from langgraph.graph import StateGraph, END

def build_vfx_graph() -> StateGraph:
    graph = StateGraph(VFXJobState)

    # Register nodes
    graph.add_node("input_guard",       input_guard_node)
    graph.add_node("nlp_planner",       nlp_planner_node)
    graph.add_node("segmentation",      segmentation_node)
    graph.add_node("depth_estimation",  depth_estimation_node)
    graph.add_node("compositing",       compositing_node)
    graph.add_node("filter",            filter_node)
    graph.add_node("error_recovery",    error_recovery_node)

    # Entry point
    graph.set_entry_point("input_guard")
    graph.add_edge("input_guard", "nlp_planner")

    # Conditional routing from planner
    graph.add_conditional_edges("nlp_planner", route_after_planner)

    # Chained edges through tool nodes
    graph.add_conditional_edges("segmentation",     route_after_segmentation)
    graph.add_conditional_edges("depth_estimation", route_after_depth)
    graph.add_conditional_edges("compositing",      route_after_compositing)
    graph.add_edge("filter",          END)
    graph.add_edge("error_recovery",  END)

    return graph.compile()
```

---

## 7. External API Integration

| Model | Provider | API Type | Used In Node | Fallback |
|-------|----------|----------|-------------|----------|
| Claude (LLM Planner) | Anthropic | REST | Node 1 | None (hard dependency) |
| SAM (Segmentation) | Replicate | REST | Node 2 | HuggingFace Inference API |
| MiDaS (Depth) | HuggingFace / Replicate | REST | Node 3 | Alternative depth model |
| SD XL + ControlNet | Replicate | REST | Node 4 | SD 1.5 + ControlNet (lower quality) |

**Cost Guard:** A pre-flight check before any Replicate/HuggingFace call estimates the cost and aborts if the estimated spend for this job exceeds $0.50 USD. This prevents runaway spend from malformed large inputs.

---

## 8. Observability & Tracing

**Structured logging** is applied at every node transition. Each log event includes:

```json
{
  "job_id":      "abc-123",
  "node":        "segmentation",
  "event":       "node_complete",
  "duration_ms": 4200,
  "model_used":  "replicate/segment-anything",
  "quality_flags": [],
  "timestamp":   "2026-06-03T10:22:01Z"
}
```

**LangSmith Integration (Optional):** If `LANGCHAIN_API_KEY` is set in the environment, all LangGraph traces are automatically forwarded to LangSmith for visual inspection. If not set, traces are written to a local `traces/` directory as newline-delimited JSON.

This dual approach means the system is observable in both a demo setting (LangSmith UI) and a local/offline setting (log files), without hard-coupling the system to a paid trace service.

---

## 9. FastAPI Server Architecture

```
app/
├── main.py               # FastAPI app, CORS, startup/shutdown lifecycle
├── routes/
│   ├── jobs.py           # POST /jobs, GET /jobs/{id}, GET /jobs/{id}/plan
│   └── results.py        # GET /jobs/{id}/result, GET /jobs/{id}/trace
├── agent/
│   ├── graph.py          # LangGraph graph construction
│   ├── nodes/
│   │   ├── input_guard.py
│   │   ├── nlp_planner.py
│   │   ├── segmentation.py
│   │   ├── depth_estimation.py
│   │   ├── compositing.py
│   │   ├── filter.py
│   │   └── error_recovery.py
│   └── state.py          # VFXJobState TypedDict
├── services/
│   ├── replicate_client.py
│   ├── anthropic_client.py
│   └── cost_guard.py
├── storage/
│   └── job_store.py      # In-memory dict for prototype; swap for Redis in prod
└── tests/
    ├── test_routing.py    # Unit tests for conditional edge logic
    └── test_nodes.py      # Integration tests with mocked API responses
```

**Job Storage:** An in-memory Python dict for the prototype (justified by 24h constraint and single-instance deployment). The interface is abstracted behind a `JobStore` class so it can be swapped for Redis or a database without changing any other code.

---

## 10. Deployment

**Local (Development):**
```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
REPLICATE_API_TOKEN=r8_...
HUGGINGFACE_API_KEY=hf_...
LANGCHAIN_API_KEY=ls__...  # Optional

docker compose up
```

**Docker Compose (Two Services):**
1. `api` — FastAPI server on port 8000
2. `worker` — Background task runner for LangGraph jobs (avoids blocking the HTTP thread)

The worker communicates with the API via an in-process queue for the prototype, which can be replaced with Celery + Redis for a production scaling path.

---

## 11. Scalability Path (Post-Evaluation)

The following upgrades can be made **without changing any node logic** because the architecture isolates each concern:

| Upgrade | What Changes | What Stays the Same |
|---------|-------------|---------------------|
| Swap SAM for a better segmentation model | `segmentation.py` API call | All other nodes, routing, state schema |
| Add video support | Input Guard node pre-processes frames; compositing node loops | NLP Planner, Depth node, API spec |
| Replace in-memory job store with Redis | `job_store.py` | All agent logic |
| Add user authentication | FastAPI middleware layer | Entire agent graph |
| Run vision models locally on GPU | Replace Replicate client with local PyTorch inference | Agent graph, routing, state schema |

This is the core architectural argument for the "Tool-Calling Orchestrator" pattern: the graph structure is the long-lived asset. The model implementations are implementation details.
