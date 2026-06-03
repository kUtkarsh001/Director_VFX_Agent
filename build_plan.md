# Build Plan — EskillVeda VFX Director Agent
**Sprint Duration:** 16–17 Hours | **Start:** Hour 0 | **Hard Stop:** Hour 17
**Reference Documents:** `prd.md` · `architecture.md` · `api_spec.md`

---

> **How to use this document**
> Work through each milestone in order. Do not skip ahead.
> Each milestone ends with a commit. Commit immediately when you hit the checkpoint — this protects your work and gives evaluators a readable git history.
> The initial prompt in Section 1 is what you paste into your coding assistant to bootstrap the project. Everything else flows from there.

---

## Section 1 — Initial Kickoff Prompt

> Copy this entire block and paste it as your first message to Claude Code (or your coding assistant of choice) before writing a single line of code. Upload `prd.md`, `architecture.md`, and `api_spec.md` alongside it.

---

```
You are a senior AI engineer helping me build the EskillVeda VFX Director Agent.
I am attaching three project documents — prd.md, architecture.md, and api_spec.md.
Read all three carefully before writing any code. Every architectural decision in
those documents has been thought through; do not redesign, simplify, or change
the structure unless I ask you to.

Here is a summary of what we are building and what I need from you right now:

PROJECT SUMMARY
---------------
An AI agent that accepts an image + free-text VFX prompt (e.g. "remove the
person and replace the sky with a sci-fi nebula") and returns a composited
output image. The agent is a LangGraph state machine. An LLM (Claude) acts as
the "Director" — it parses the prompt and decides which specialist models to
call: SAM for segmentation, MiDaS for depth estimation, and Stable Diffusion
XL + ControlNet for compositing. Models are called via Replicate/HuggingFace
APIs (not run locally). A FastAPI server exposes the agent as a REST API.

KEY CONSTRAINTS
---------------
- Python 3.11
- LangGraph (latest) for the state machine
- Claude claude-sonnet-4-20250514 for the NLP Planner node
- All vision models called via Replicate REST API
- No frontend, no auth, no database — in-memory job store only
- Each node must be a separate file under app/agent/nodes/
- Node functions must be < 80 lines each

EXACT FOLDER STRUCTURE TO SCAFFOLD (from architecture.md section 9)
---------------------------------------------------------------------
vfx-director-agent/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── jobs.py
│   │   └── results.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── graph.py
│   │   ├── state.py
│   │   └── nodes/
│   │       ├── __init__.py
│   │       ├── input_guard.py
│   │       ├── nlp_planner.py
│   │       ├── segmentation.py
│   │       ├── depth_estimation.py
│   │       ├── compositing.py
│   │       ├── filter.py
│   │       └── error_recovery.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── anthropic_client.py
│   │   ├── replicate_client.py
│   │   └── cost_guard.py
│   └── storage/
│       ├── __init__.py
│       └── job_store.py
├── tests/
│   ├── test_routing.py
│   └── test_nodes.py
├── traces/              # Local trace output directory
├── .env.example
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md

FIRST TASK — DO THIS NOW
------------------------
1. Create the complete folder structure above with empty __init__.py files.
2. Write requirements.txt with these exact packages:
   fastapi==0.111.0
   uvicorn[standard]==0.29.0
   langgraph>=0.1.0
   langchain-anthropic>=0.1.0
   anthropic>=0.28.0
   replicate>=0.28.0
   httpx>=0.27.0
   pillow>=10.3.0
   python-multipart>=0.0.9
   pydantic>=2.7.0
   python-dotenv>=1.0.0
   pytest>=8.0.0
   pytest-asyncio>=0.23.0
3. Write .env.example with these keys (no values):
   ANTHROPIC_API_KEY=
   REPLICATE_API_TOKEN=
   HUGGINGFACE_API_KEY=
   LANGCHAIN_API_KEY=
   MAX_COST_PER_JOB_USD=0.50
4. Write app/agent/state.py with the exact VFXJobState TypedDict from
   architecture.md section 3. Do not simplify it.
5. Write app/agent/graph.py with the full LangGraph graph construction from
   architecture.md section 6, but with all node functions imported as stubs
   (each stub just returns the state unchanged and prints its name).
6. Verify the graph compiles by running: python -c "from app.agent.graph import build_vfx_graph; g = build_vfx_graph(); print('Graph OK')"

Do not implement any node logic yet. Just scaffold and verify the graph compiles.
Tell me when step 6 passes.
```

---

## Section 2 — Pre-Build Checklist

Complete these before starting the timer. They are not part of the 17-hour sprint.

- [ ] Create GitHub repo: `vfx-director-agent` (private or public)
- [ ] Clone repo locally
- [ ] Create Python 3.11 virtual environment: `python3.11 -m venv .venv && source .venv/bin/activate`
- [ ] Get API keys and store them in `.env` (never commit this file):
  - Anthropic: [console.anthropic.com](https://console.anthropic.com)
  - Replicate: [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)
  - HuggingFace: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- [ ] Add `.env` and `__pycache__` to `.gitignore`
- [ ] Confirm: `python --version` prints `3.11.x`

---

## Section 3 — Milestone-by-Milestone Build Plan

---

### MILESTONE 0 — Scaffold & State Schema
**⏱ Hours 0:00 – 0:45**

**Goal:** Repo is initialised. Folder structure exists. `VFXJobState` is written. LangGraph graph compiles with stub nodes.

**Paste the kickoff prompt from Section 1.** Work with your coding assistant until `python -c "from app.agent.graph import build_vfx_graph; ..."` prints `Graph OK`.

**Key things to verify before committing:**
- `app/agent/state.py` has every field listed in `architecture.md` Section 3: `job_id`, `original_image`, `image_format`, `user_prompt`, `extracted_intent`, `execution_plan`, `plan_confidence`, `masks`, `mask_confidence`, `depth_map`, `final_image`, `nodes_executed`, `errors`, `quality_flags`, `status`
- All routing functions (`route_after_planner`, `route_after_segmentation`, `route_after_depth`, `route_after_compositing`) exist in `graph.py` even if they just return a hardcoded string for now
- `requirements.txt` is committed

```
✅ COMMIT 1
git add .
git commit -m "chore: initialise project scaffold with folder structure, state schema, and stub LangGraph nodes"
```

---

### MILESTONE 1 — Input Guard Node
**⏱ Hours 0:45 – 1:45**

**Goal:** `app/agent/nodes/input_guard.py` fully implemented and tested.

**Prompt to your assistant:**
```
Implement app/agent/nodes/input_guard.py based on the spec in architecture.md
section 4.1. The node must:
1. Validate image format — accept only jpeg, png, webp. Reject anything else with
   error_code INPUT_INVALID_FORMAT and set state["status"] = "failed".
2. Validate image size — reject if > 10 MB with error_code INPUT_TOO_LARGE.
3. Validate prompt — reject if empty or > 500 chars with error_code PROMPT_INVALID.
4. Resize image if either dimension > 2048px using Pillow, preserving aspect ratio.
5. Store the (possibly resized) image as bytes back in state["original_image"].
6. Set state["status"] = "planning" on success.
7. Append to state["nodes_executed"].
The function signature must be: def input_guard_node(state: VFXJobState) -> VFXJobState
Keep the file under 80 lines.
```

**Write a quick manual test** — create `tests/test_nodes.py` with one test that passes a small JPEG in and asserts `status == "planning"`, and one that passes a 600-char prompt and asserts `status == "failed"`.

```
✅ COMMIT 2
git add .
git commit -m "feat(nodes): implement input_guard with image validation, size check, and prompt validation"
```

---

### MILESTONE 2 — Anthropic Service Client
**⏱ Hours 1:45 – 2:30**

**Goal:** `app/services/anthropic_client.py` wraps the Anthropic SDK. The NLP planner node will call this.

**Prompt to your assistant:**
```
Implement app/services/anthropic_client.py. It must expose a single async function:

async def call_planner(image_base64: str, prompt: str) -> dict

The function calls claude-sonnet-4-20250514 with temperature=0.1.
The system prompt instructs Claude to return ONLY a JSON object with these fields:
  target_objects (list[str])
  actions (list[str])
  required_nodes (list[str]) — values must be a subset of:
    ["segmentation", "depth_estimation", "compositing", "filter"]
  plan_summary (str)
  confidence (float 0.0–1.0)
  clarification_needed (bool)
  clarification_question (str | null)

The user message combines the VFX prompt with a base64-encoded image (vision input).
Strip any ```json fences before calling json.loads().
Raise a ValueError if the response cannot be parsed.
Load ANTHROPIC_API_KEY from environment via python-dotenv.
```

**Do not test against the live API yet** — write a mock test that patches the Anthropic client and verifies the JSON parsing and stripping logic.

```
✅ COMMIT 3
git add .
git commit -m "feat(services): add Anthropic client wrapper with JSON schema extraction and fence stripping"
```

---

### MILESTONE 3 — NLP Planner Node
**⏱ Hours 2:30 – 4:00**

**Goal:** `app/agent/nodes/nlp_planner.py` implemented. This is the cognitive core of the agent — take time to get it right.

**Prompt to your assistant:**
```
Implement app/agent/nodes/nlp_planner.py. It calls app/services/anthropic_client.call_planner().
After receiving the parsed JSON response:
1. Store the result in state["extracted_intent"] (the full dict).
2. Store "plan_summary" string in state["execution_plan"].
3. Store "confidence" float in state["plan_confidence"].
4. If confidence < 0.7 or clarification_needed is true:
   - Set state["status"] = "clarification_required"
   - Do NOT update nodes_executed (the plan did not execute)
   - Return early
5. Otherwise set state["status"] = "running".
6. Append "nlp_planner" to state["nodes_executed"].
7. Emit a structured log line to stdout as JSON:
   {"job_id": ..., "node": "nlp_planner", "event": "complete",
    "confidence": ..., "required_nodes": ..., "timestamp": "..."}

Also update graph.py: replace the nlp_planner stub with the real import.
Update route_after_planner() to handle the "clarification_required" status:
  if state["status"] == "clarification_required": return END
```

**Live test this node** — call it manually with a real image and prompt. Verify the JSON comes back with the right fields.

```
✅ COMMIT 4
git add .
git commit -m "feat(nodes): implement nlp_planner with Claude vision call, confidence guard, and plan extraction"
```

---

### MILESTONE 4 — FastAPI Server + Job Store + Core Endpoints
**⏱ Hours 4:00 – 5:30**

**Goal:** `POST /jobs` and `GET /jobs/{id}` are working. You can submit a job and poll its status.

**Prompt to your assistant:**
```
Implement three files:

1. app/storage/job_store.py
   A JobStore class backed by a plain Python dict (thread-safe with asyncio.Lock).
   Methods:
     create_job(job_id, initial_state) -> None
     get_job(job_id) -> VFXJobState | None
     update_job(job_id, partial_state: dict) -> None
     list_jobs() -> list[str]
   This is the only place job state is stored. All other code uses this class.

2. app/routes/jobs.py
   POST /jobs:
     - Accept multipart/form-data: image (UploadFile), prompt (str),
       response_format (str = "url"), confirm_plan (bool = False)
     - Validate with Input Guard inline (call input_guard_node directly before
       dispatching to LangGraph)
     - Generate a UUID job_id
     - Store initial state in JobStore with status="queued"
     - Run the LangGraph agent in a background task (FastAPI BackgroundTasks)
     - Return 202 with job_id, status, plan_url, status_url
       (match the response schema exactly from api_spec.md section POST /jobs)
   GET /jobs/{job_id}:
     - Look up job in JobStore
     - Return 404 if not found
     - Return the full status response from api_spec.md section GET /jobs/{id}
     - Include current_node, nodes_completed, quality_flags, errors

3. app/main.py
   FastAPI app with CORS, lifespan events, and the jobs router mounted.
   Run with: uvicorn app.main:app --reload --port 8000

Use the exact response envelope from api_spec.md:
  {"success": true/false, "data": {...}, "error": null/{...}}
```

**Test manually:**
```bash
curl -X POST http://localhost:8000/jobs \
  -F "image=@test.jpg" \
  -F "prompt=make it black and white"
# Should return {"success": true, "data": {"job_id": "...", "status": "queued"}}
```

```
✅ COMMIT 5
git add .
git commit -m "feat(api): add FastAPI server with POST /jobs and GET /jobs/{id}, BackgroundTasks runner, and in-memory JobStore"
```

---

### MILESTONE 5 — Replicate Service Client
**⏱ Hours 5:30 – 6:15**

**Goal:** `app/services/replicate_client.py` is a reusable wrapper for all Replicate API calls. Every downstream node calls this — it pays to build it cleanly once.

**Prompt to your assistant:**
```
Implement app/services/replicate_client.py. It must expose:

async def run_model(model_version: str, input_payload: dict) -> dict

It uses the Replicate Python SDK (import replicate).
Load REPLICATE_API_TOKEN from environment.
The function should:
1. Call replicate.async_run(model_version, input=input_payload)
2. Wait for the result (Replicate returns a Future)
3. Return the result dict
4. On HTTP timeout (> 60s), raise a TimeoutError with message "Replicate API timeout"
5. On any Replicate API error, raise a RuntimeError with the error message

Also create the three Replicate model version constants at the top of the file:
SAM_MODEL = "schananas/grounded_sam:..."        # fill in latest version
MIDAS_MODEL = "andreasjansson/midas:..."        # fill in latest version
SDXL_CONTROLNET_MODEL = "..."                   # fill in latest depth-conditioned version

Look up the current model versions on replicate.com before hardcoding them.
```

> **Note:** Spend 5 minutes on [replicate.com](https://replicate.com) to verify the correct model slugs and latest version hashes before writing the constants.

```
✅ COMMIT 6
git add .
git commit -m "feat(services): add Replicate async client wrapper with timeout handling and model version constants"
```

---

### MILESTONE 6 — Segmentation Node (SAM)
**⏱ Hours 6:15 – 8:00**

**Goal:** `app/agent/nodes/segmentation.py` calls SAM via Replicate, processes masks, sets quality flags.

**Prompt to your assistant:**
```
Implement app/agent/nodes/segmentation.py based on architecture.md section 4.3.

The node must:
1. Check: if "segmentation" not in state["extracted_intent"]["required_nodes"], return state unchanged.
2. Get target_objects from state["extracted_intent"]["target_objects"].
3. Call run_model(SAM_MODEL, {image: <base64_image>, labels: target_objects}).
4. SAM returns a list of mask dicts. Extract the binary mask bytes for each object.
5. Store in state["masks"] as {object_name: mask_bytes}.
6. Compute mask_confidence: use the highest logit score from SAM's output as a proxy.
   If SAM doesn't return logits, default to 0.8 (assume success unless the mask
   area is suspiciously small < 1% of image area, in which case use 0.4).
7. If mask_confidence < 0.6:
   - Append "low_confidence_mask" to state["quality_flags"]
   - Do NOT fail — continue.
8. On any exception from run_model:
   - Append to state["errors"]: {node, error_code, message, timestamp}
   - Set a flag in state so the error_recovery node is triggered
9. Append "segmentation" to state["nodes_executed"].
10. Emit a structured log line to stdout.

Update route_after_segmentation() in graph.py:
  - if state["errors"]: return "error_recovery"
  - if "depth_estimation" in required_nodes: return "depth_estimation"
  - if "compositing" in required_nodes: return "compositing"
  - else: return END
```

**Test prompt:** `"isolate the cat"` on a photo with a cat. Verify `state["masks"]` is populated.

```
✅ COMMIT 7
git add .
git commit -m "feat(nodes): implement segmentation node with SAM via Replicate, mask confidence scoring, and quality flags"
```

---

### MILESTONE 7 — Depth Estimation Node (MiDaS)
**⏱ Hours 8:00 – 9:30**

**Goal:** `app/agent/nodes/depth_estimation.py` calls MiDaS via Replicate/HuggingFace, stores a depth map.

**Prompt to your assistant:**
```
Implement app/agent/nodes/depth_estimation.py based on architecture.md section 4.4.

The node must:
1. Check: if "depth_estimation" not in required_nodes, return state unchanged.
2. Call run_model(MIDAS_MODEL, {image: <base64_image>}).
3. MiDaS returns a URL to the depth map image. Fetch it with httpx.get() and
   store the raw bytes in state["depth_map"].
4. On any exception:
   - Append to state["errors"]: {node: "depth_estimation", error_code: "DEPTH_FAILED", ...}
   - Append "depth_estimation_failed" to state["quality_flags"]
   - Do NOT abort — the compositing node can run without depth conditioning
     (it will fall back to standard inpainting). This is a degraded but valid result.
5. Append "depth_estimation" to state["nodes_executed"].
6. Emit a structured log line.

Update route_after_depth() in graph.py:
  - if any new errors from this node: return "error_recovery" only if
    compositing is not in required_nodes (depth failure is non-fatal)
  - if "compositing" in required_nodes: return "compositing"
  - else: return END
```

**Test prompt:** any image. Verify `state["depth_map"]` contains bytes.

```
✅ COMMIT 8
git add .
git commit -m "feat(nodes): implement depth_estimation node using MiDaS via Replicate with graceful degradation on failure"
```

---

### MILESTONE 8 — Compositing Node (SD-XL + ControlNet)
**⏱ Hours 9:30 – 12:30**

**This is the hardest milestone. Budget 3 hours. Do not rush it.**

**Goal:** `app/agent/nodes/compositing.py` produces the final composited image using Stable Diffusion XL + ControlNet depth conditioning.

**Prompt to your assistant:**
```
Implement app/agent/nodes/compositing.py based on architecture.md section 4.5.
This is the most complex node. Read the spec carefully.

The node must:
1. Check: if "compositing" not in required_nodes, return state unchanged.

2. Build the compositing prompt:
   - Extract the generative instruction from state["extracted_intent"]["actions"]
     and state["extracted_intent"]["target_objects"]
   - Construct a string like: "sci-fi nebula sky, dramatic lighting, cinematic"
   - Append negative prompt: "low quality, blurry, distorted, artifacts"

3. Build the inpainting mask:
   - If state["masks"] is populated, take the union of all masks using Pillow
     (combine multiple mask images into one binary mask, white = area to replace)
   - If state["masks"] is None, create a full-image mask (replace everything)

4. Build the input payload for SDXL_CONTROLNET_MODEL:
   - image: base64-encoded original image
   - mask: base64-encoded combined mask
   - prompt: the generative instruction from step 2
   - negative_prompt: the negative string
   - If state["depth_map"] is not None:
       controlnet_conditioning_image: base64-encoded depth map
       controlnet_conditioning_scale: 0.8
   - Else (depth failed — use standard inpainting without depth conditioning):
       omit the controlnet fields

5. Call run_model(SDXL_CONTROLNET_MODEL, payload).

6. The model returns a URL to the output image. Fetch with httpx.get() and
   store bytes in state["final_image"].
   Also store state["image_format"] = "png".

7. Set state["status"] = "done".

8. On any exception: append to errors, set status = "failed" only if
   no usable fallback is available.

9. Append "compositing" to nodes_executed. Emit structured log.

Update route_after_compositing() in graph.py:
  - if state["status"] == "done": return END
  - if state["errors"]: return "error_recovery"
  - return END
```

**This is where you should run an end-to-end test for the first time.** Submit a real job via `POST /jobs` with a photo and a compound prompt. Watch the logs. The full pipeline (input_guard → nlp_planner → segmentation → depth_estimation → compositing) should run.

If the Replicate model slugs are wrong, fix them now.

```
✅ COMMIT 9
git add .
git commit -m "feat(nodes): implement compositing node with SD-XL + ControlNet depth conditioning and mask union"

✅ COMMIT 10  (after successful end-to-end test)
git add .
git commit -m "fix: resolve end-to-end integration issues from first full pipeline test"
```

---

### MILESTONE 9 — Error Recovery Node + Retry Logic
**⏱ Hours 12:30 – 13:45**

**Goal:** `app/agent/nodes/error_recovery.py` centralises all failure handling. One retry on network errors.

**Prompt to your assistant:**
```
Implement app/agent/nodes/error_recovery.py based on architecture.md section 4.7.

The node must:
1. Read the most recent error from state["errors"] (last item in list).
2. Determine if it is retryable:
   Retryable error codes: API_TIMEOUT, HTTP_503, HTTP_502, CONNECTION_ERROR
   Non-retryable: INPUT_INVALID_FORMAT, PROMPT_INVALID, COST_GUARD_EXCEEDED,
                  any error not in the retryable list
3. Check state for a "_retry_count" key (store it in errors metadata or add a
   new field retry_count: int = 0 to VFXJobState — your choice, be consistent).
4. If retryable AND retry_count < 1:
   - Increment retry_count
   - Return the state with the failed node's name stored in a "_retry_target" field
   - The graph will re-route to that node (update graph.py edges accordingly)
5. Otherwise:
   - Set state["status"] = "failed"
   - Emit a structured error log line
   - Return END
6. Append "error_recovery" to nodes_executed.

Add retry_count: int and retry_target: Optional[str] to VFXJobState in state.py.
Update graph.py: error_recovery node gets a conditional edge:
  - if retry_target is set and retry_count <= 1: return retry_target node name
  - else: return END
```

```
✅ COMMIT 11
git add .
git commit -m "feat(nodes): add error_recovery node with single-retry logic for transient API failures"
```

---

### MILESTONE 10 — Cost Guard
**⏱ Hours 13:45 – 14:30**

**Goal:** `app/services/cost_guard.py` runs before any Replicate API call. Protects against runaway spend.

**Prompt to your assistant:**
```
Implement app/services/cost_guard.py.

It must expose one function:
  def check_cost(required_nodes: list[str], image_size_mb: float) -> None

The function estimates the cost of the job and raises a CostGuardException
if the estimate exceeds MAX_COST_PER_JOB_USD (from env, default 0.50).

Cost table (approximate, hard-coded):
  segmentation: $0.04 per call
  depth_estimation: $0.02 per call
  compositing: $0.08 per call
  filter: $0.00

Image size multiplier: if image_size_mb > 5: multiply total cost by 1.5

CostGuardException should extend Exception and include:
  estimated_cost, limit, message

In nlp_planner.py, after the plan is extracted and before returning:
  call check_cost(required_nodes, image_size_mb)
  catch CostGuardException and:
    - append to state["errors"] with error_code "COST_GUARD_EXCEEDED"
    - set state["status"] = "failed"
    - return early
```

```
✅ COMMIT 12
git add .
git commit -m "feat(services): add cost_guard with per-job spend estimation before Replicate API calls"
```

---

### MILESTONE 11 — Result, Plan, and Trace Endpoints
**⏱ Hours 14:30 – 15:45**

**Goal:** All remaining API endpoints from `api_spec.md` are implemented. The API is complete.

**Prompt to your assistant:**
```
Implement app/routes/results.py with four endpoints. Match api_spec.md exactly.

GET /jobs/{job_id}/plan
  - Return 409 PLAN_NOT_READY if status is "queued"
  - Return the plan object from state["extracted_intent"] and state["execution_plan"]
  - Include confidence and estimated fields from api_spec.md response schema

POST /jobs/{job_id}/confirm
  - Accept JSON body {"proceed": bool}
  - Return 409 INVALID_STATE_TRANSITION if status != "awaiting_confirmation"
  - If proceed=true: set status="running", resume graph execution (this requires
    you to store the paused graph state — for the prototype, just re-run the graph
    from the segmentation step using the existing planned state)
  - If proceed=false: set status="failed" with message "Cancelled by user"

GET /jobs/{job_id}/result
  - Return 409 RESULT_NOT_READY if status != "done"
  - Accept ?format=url or ?format=base64
  - For base64: return state["final_image"] as base64-encoded string in JSON
  - For url (default): write the image to traces/{job_id}_result.png and
    return a file URL. Use FileResponse or a static files mount.
  - Include quality_flags in the response

GET /jobs/{job_id}/trace
  - Return the full trace from state["nodes_executed"] and per-node metadata
  - Build the trace response shape from api_spec.md section GET /jobs/{id}/trace
  - Include duration_ms per node (store node start/end times in a
    new field node_timings: dict in VFXJobState — add it to state.py)

Mount the results router in main.py.
```

**Test all six endpoints** with curl. Verify every response matches the envelope: `{"success": true/false, "data": {...}, "error": null/{...}}`

```
✅ COMMIT 13
git add .
git commit -m "feat(api): implement /plan, /confirm, /result, and /trace endpoints; complete REST API surface"
```

---

### MILESTONE 12 — Filter Node (No-API Path)
**⏱ Hours 15:45 – 16:15**

**Goal:** `app/agent/nodes/filter.py` handles simple operations (greyscale, contrast) using only Pillow. Zero external API calls.

**Prompt to your assistant:**
```
Implement app/agent/nodes/filter.py based on prd.md FR-10 and architecture.md section 4.6.

Supported operations (detect from state["extracted_intent"]["actions"]):
  "greyscale" or "black_and_white" → convert to L mode, back to RGB
  "increase_contrast"              → ImageEnhance.Contrast(img).enhance(1.5)
  "decrease_contrast"              → ImageEnhance.Contrast(img).enhance(0.6)
  "brighten"                       → ImageEnhance.Brightness(img).enhance(1.3)
  "darken"                         → ImageEnhance.Brightness(img).enhance(0.7)
  default (unrecognised action)    → return image unchanged, add "filter_action_unknown"
                                     to quality_flags

After applying the filter:
  - Store result bytes in state["final_image"]
  - Set state["status"] = "done"
  - Append "filter" to nodes_executed
  - Emit structured log line

This node must never fail — it uses only Pillow, which is already installed.
Keep it under 60 lines.
```

**Test with prompt:** `"make it black and white"`. Verify it routes to the filter node and skips all Replicate calls.

```
✅ COMMIT 14
git add .
git commit -m "feat(nodes): add filter node for greyscale and contrast ops using Pillow only (zero API cost)"
```

---

### MILESTONE 13 — Docker Compose
**⏱ Hours 16:15 – 16:45**

**Goal:** `docker compose up` starts the server. Someone cloning the repo can run the project in one command.

**Prompt to your assistant:**
```
Write Dockerfile and docker-compose.yml.

Dockerfile:
  FROM python:3.11-slim
  WORKDIR /app
  COPY requirements.txt .
  RUN pip install --no-cache-dir -r requirements.txt
  COPY . .
  EXPOSE 8000
  CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

docker-compose.yml:
  version: "3.9"
  services:
    api:
      build: .
      ports:
        - "8000:8000"
      env_file:
        - .env
      volumes:
        - ./traces:/app/traces
      restart: unless-stopped

Also ensure traces/ directory has a .gitkeep file so it's tracked by git
but its contents are gitignored. Add traces/*.png and traces/*.json to .gitignore.
```

```
✅ COMMIT 15
git add .
git commit -m "chore(docker): add Dockerfile and docker-compose for single-command local deployment"
```

---

### MILESTONE 14 — README + Final Polish
**⏱ Hours 16:45 – 17:00**

**Goal:** A reviewer can clone the repo, read the README, and understand the project in 3 minutes.

**Prompt to your assistant:**
```
Write README.md. It must include:

1. One-sentence description of what the project does
2. Architecture diagram (ASCII, copy from architecture.md section 2 exactly)
3. Tech stack table (LangGraph, Claude, SAM, MiDaS, SD-XL + ControlNet, FastAPI)
4. Quick start:
   git clone ...
   cp .env.example .env   # fill in your keys
   docker compose up
5. Example usage — three curl commands:
   - Submit a job
   - Poll status
   - Get result
6. Six-row table: each node name, what it does, which model it calls
7. One paragraph on why this is an agent and not a script
   (reference the conditional routing logic — different prompts skip different nodes)
8. Link to prd.md, architecture.md, api_spec.md in a "Design Documents" section
```

```
✅ COMMIT 16 — FINAL COMMIT
git add .
git commit -m "docs: add README with architecture overview, quick start guide, and example curl commands"

git tag v1.0.0
git push origin main --tags
```

---

## Section 4 — Commit History Summary

The following is the complete, ordered commit log for the project. This is what reviewers will see.

```
COMMIT 1  chore: initialise project scaffold with folder structure, state schema, and stub LangGraph nodes
COMMIT 2  feat(nodes): implement input_guard with image validation, size check, and prompt validation
COMMIT 3  feat(services): add Anthropic client wrapper with JSON schema extraction and fence stripping
COMMIT 4  feat(nodes): implement nlp_planner with Claude vision call, confidence guard, and plan extraction
COMMIT 5  feat(api): add FastAPI server with POST /jobs and GET /jobs/{id}, BackgroundTasks runner, and in-memory JobStore
COMMIT 6  feat(services): add Replicate async client wrapper with timeout handling and model version constants
COMMIT 7  feat(nodes): implement segmentation node with SAM via Replicate, mask confidence scoring, and quality flags
COMMIT 8  feat(nodes): implement depth_estimation node using MiDaS via Replicate with graceful degradation on failure
COMMIT 9  feat(nodes): implement compositing node with SD-XL + ControlNet depth conditioning and mask union
COMMIT 10 fix: resolve end-to-end integration issues from first full pipeline test
COMMIT 11 feat(nodes): add error_recovery node with single-retry logic for transient API failures
COMMIT 12 feat(services): add cost_guard with per-job spend estimation before Replicate API calls
COMMIT 13 feat(api): implement /plan, /confirm, /result, and /trace endpoints; complete REST API surface
COMMIT 14 feat(nodes): add filter node for greyscale and contrast ops using Pillow only (zero API cost)
COMMIT 15 chore(docker): add Dockerfile and docker-compose for single-command local deployment
COMMIT 16 docs: add README with architecture overview, quick start guide, and example curl commands
```

---

## Section 5 — Time Budget Summary

| Milestone | Task | Hours |
|-----------|------|-------|
| 0 | Scaffold + State Schema | 0:45 |
| 1 | Input Guard Node | 1:00 |
| 2 | Anthropic Service Client | 0:45 |
| 3 | NLP Planner Node | 1:30 |
| 4 | FastAPI + Job Store + Core Endpoints | 1:30 |
| 5 | Replicate Service Client | 0:45 |
| 6 | Segmentation Node (SAM) | 1:45 |
| 7 | Depth Estimation Node (MiDaS) | 1:30 |
| 8 | Compositing Node (SD-XL + ControlNet) | 3:00 |
| 9 | Error Recovery + Retry | 1:15 |
| 10 | Cost Guard | 0:45 |
| 11 | Plan, Result, Trace Endpoints | 1:15 |
| 12 | Filter Node | 0:30 |
| 13 | Docker Compose | 0:30 |
| 14 | README + Final Polish | 0:15 |
| **Total** | | **~16:45** |

---

## Section 6 — Risk Register & Contingency

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Replicate model slug is outdated | High | Milestone 8 blocked | Spend 5 min on replicate.com before Milestone 5 |
| SAM returns unexpected output format | Medium | Milestone 6 delayed | Read Replicate docs for the SAM model before coding |
| SD-XL + ControlNet depth endpoint not available | Medium | Milestone 8 degrades | Fall back to SD 1.5 + ControlNet; add quality_flag |
| Compositing node takes >3 hours | Low | Schedule slips | Cut `/confirm` endpoint and filters; these are "Should Have" |
| LangGraph version API mismatch | Low | Graph fails to compile | Pin `langgraph==0.1.19` in requirements.txt if you hit this |

**If you fall behind after Milestone 8:** skip Milestone 9 (error recovery), 10 (cost guard), and 12 (filter). These are all "Should Have". The core agent will still be fully functional and impressive. Commit what you have, write "Stretch goals not implemented" in README, and move on.

---

## Section 7 — What the Evaluator Will See

A git history with 16 clean commits, each focused on exactly one thing. A `README.md` that explains the architecture clearly. A working FastAPI server that demonstrates the agent pipeline. Node files that are each < 80 lines and independently readable. A `VFXJobState` TypedDict that makes the data flow between nodes self-documenting. And a system that gives a different execution trace for `"make it greyscale"` versus `"remove the background and replace the sky"` — which is the entire point of building an agent instead of a script.
