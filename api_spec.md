# API Specification
## EskillVeda VFX Director Agent
**Version:** 1.1 | **Base URL:** `http://localhost:8000` | **Date:** 2026-06-03

---

## Overview

The VFX Director Agent exposes a simple asynchronous REST API. Jobs are submitted, processed in the background by the LangGraph agent, and polled by the client until completion.

**Design decisions:**
- Jobs are asynchronous because vision model APIs have latencies of 10–60s each. Blocking HTTP requests for 90s is unreliable.
- Plan preview is exposed as a separate endpoint so callers can verify the agent's interpretation before the expensive tool calls begin.
- All responses are JSON. Images are returned as base64-encoded strings or as direct download URLs (configurable via `response_format`).

---

## Authentication

For the prototype, no authentication is required. All endpoints are open on localhost. In a production deployment, an `Authorization: Bearer <token>` header would be required — the FastAPI middleware layer handles this without touching agent logic.

---

## Common Response Envelope

All responses follow a consistent envelope:

```json
{
  "success": true,
  "data":    { ... },
  "error":   null
}
```

On error:
```json
{
  "success": false,
  "data":    null,
  "error": {
    "code":    "SEGMENTATION_FAILED",
    "message": "SAM API returned a 503 after 1 retry.",
    "node":    "segmentation",
    "job_id":  "abc-123"
  }
}
```

---

## Endpoints

---

### POST /jobs
**Submit a new VFX job.**

Accepts an image file and a text prompt. Returns immediately with a `job_id`. Processing begins asynchronously.

#### Request
```
Content-Type: multipart/form-data
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File (JPEG/PNG/WebP) | Yes | The source image. Max 10 MB. |
| `prompt` | string | Yes | Free-text VFX instruction. Max 500 chars. |
| `response_format` | string | No | `"url"` (default) or `"base64"`. Controls how output image is returned. |
| `confirm_plan` | boolean | No | If `true`, job pauses after planning and waits for `/confirm` before executing tools. Default: `false`. |

#### Example Request
```bash
curl -X POST http://localhost:8000/jobs \
  -F "image=@portrait.jpg" \
  -F "prompt=Remove the person from the background and replace the sky with a dramatic sci-fi nebula" \
  -F "confirm_plan=true"
```

#### Response — 202 Accepted
```json
{
  "success": true,
  "data": {
    "job_id":    "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "status":    "queued",
    "created_at": "2026-06-03T10:20:00Z",
    "plan_url":  "http://localhost:8000/jobs/7f3a2c91/plan",
    "status_url": "http://localhost:8000/jobs/7f3a2c91"
  },
  "error": null
}
```

#### Response — 400 Bad Request (Input Validation Failure)
```json
{
  "success": false,
  "data":    null,
  "error": {
    "code":    "INVALID_INPUT",
    "message": "Image file exceeds 10 MB limit. Received: 14.2 MB.",
    "node":    "input_guard",
    "job_id":  null
  }
}
```

---

### GET /jobs/{job_id}
**Get the current status of a job.**

Poll this endpoint until `status` is `"done"` or `"failed"`. Recommended polling interval: 3–5 seconds.

#### Path Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | UUID string | The job identifier returned by POST /jobs. |

#### Example Request
```bash
curl http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44
```

#### Response — 200 OK (Job In Progress)
```json
{
  "success": true,
  "data": {
    "job_id":          "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "status":          "running",
    "current_node":    "depth_estimation",
    "nodes_completed": ["input_guard", "nlp_planner", "segmentation"],
    "created_at":      "2026-06-03T10:20:00Z",
    "updated_at":      "2026-06-03T10:20:22Z",
    "quality_flags":   [],
    "errors":          []
  },
  "error": null
}
```

#### Response — 200 OK (Job Done)
```json
{
  "success": true,
  "data": {
    "job_id":          "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "status":          "done",
    "current_node":    null,
    "nodes_completed": ["input_guard", "nlp_planner", "segmentation", "depth_estimation", "compositing"],
    "created_at":      "2026-06-03T10:20:00Z",
    "updated_at":      "2026-06-03T10:21:47Z",
    "quality_flags":   ["low_confidence_mask"],
    "errors":          [],
    "result_url":      "http://localhost:8000/jobs/7f3a2c91/result",
    "trace_url":       "http://localhost:8000/jobs/7f3a2c91/trace"
  },
  "error": null
}
```

#### Response — 200 OK (Job Failed)
```json
{
  "success": true,
  "data": {
    "job_id":  "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "status":  "failed",
    "errors": [
      {
        "node":       "compositing",
        "error_code": "API_TIMEOUT",
        "message":    "Replicate API did not respond within 60s after 1 retry.",
        "timestamp":  "2026-06-03T10:21:05Z"
      }
    ]
  },
  "error": null
}
```

#### Status Values
| Status | Meaning |
|--------|---------|
| `queued` | Job received, not yet started |
| `planning` | NLP Planner is running |
| `awaiting_confirmation` | Plan ready, waiting for POST /jobs/{id}/confirm |
| `running` | Tool nodes are executing |
| `done` | Job completed successfully |
| `failed` | Job failed; see `errors` field |

---

### GET /jobs/{job_id}/plan
**Retrieve the agent's execution plan.**

Available once the NLP Planner node has completed (status ≥ `"running"` or `"awaiting_confirmation"`). Returns the agent's interpretation of the prompt and the planned execution sequence.

#### Example Request
```bash
curl http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44/plan
```

#### Response — 200 OK
```json
{
  "success": true,
  "data": {
    "job_id":    "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "prompt":    "Remove the person from the background and replace the sky with a dramatic sci-fi nebula",
    "plan": {
      "target_objects":  ["person", "sky"],
      "actions":         ["remove_object", "replace_background"],
      "required_nodes":  ["segmentation", "depth_estimation", "compositing"],
      "plan_summary":    "Step 1: Use SAM to create separate alpha masks for the 'person' and 'sky' regions. Step 2: Run MiDaS depth estimation on the original frame to extract scene geometry. Step 3: Feed the sky mask, depth map, and the prompt 'dramatic sci-fi nebula' to Stable Diffusion XL + ControlNet for depth-conditioned inpainting.",
      "confidence":      0.94,
      "estimated_cost_usd": 0.08,
      "estimated_duration_seconds": 75
    }
  },
  "error": null
}
```

#### Response — 409 Conflict (Plan Not Ready)
```json
{
  "success": false,
  "data":    null,
  "error": {
    "code":    "PLAN_NOT_READY",
    "message": "The NLP Planner has not yet completed for this job. Current status: queued.",
    "node":    null,
    "job_id":  "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44"
  }
}
```

---

### POST /jobs/{job_id}/confirm
**Confirm execution of the plan (only required when `confirm_plan: true` was set).**

If the job was submitted with `confirm_plan: true`, it will pause in `awaiting_confirmation` status after planning. Call this endpoint to approve the plan and begin tool execution. Alternatively, call with `{"proceed": false}` to cancel the job.

#### Request Body
```json
{ "proceed": true }
```

#### Example Request
```bash
curl -X POST http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44/confirm \
  -H "Content-Type: application/json" \
  -d '{"proceed": true}'
```

#### Response — 200 OK
```json
{
  "success": true,
  "data": {
    "job_id": "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "status": "running",
    "message": "Execution confirmed. Job is now running."
  },
  "error": null
}
```

#### Response — 409 Conflict (Wrong State)
```json
{
  "success": false,
  "data":    null,
  "error": {
    "code":    "INVALID_STATE_TRANSITION",
    "message": "Job is not in 'awaiting_confirmation' state. Current status: running.",
    "job_id":  "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44"
  }
}
```

---

### GET /jobs/{job_id}/result
**Download the composited output image.**

Only available when `status == "done"`.

#### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `"url"` | `"url"` returns a redirect to the image file. `"base64"` embeds the image in JSON. |

#### Example Request (URL redirect)
```bash
curl -L http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44/result
# Redirects to temporary download URL; saves image to disk
```

#### Example Request (Base64)
```bash
curl "http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44/result?format=base64"
```

#### Response — 200 OK (Base64)
```json
{
  "success": true,
  "data": {
    "job_id":         "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "image_base64":   "iVBORw0KGgoAAAANSUhEUgAA...",
    "image_format":   "png",
    "width_px":       1920,
    "height_px":      1080,
    "quality_flags":  ["low_confidence_mask"],
    "expires_at":     "2026-06-04T10:21:47Z"
  },
  "error": null
}
```

#### Note on `quality_flags`
If `quality_flags` contains entries, the result image is returned but the caller should surface these to the user:

| Flag | Meaning |
|------|---------|
| `low_confidence_mask` | SAM's object mask had IoU < 0.6; edges may be imprecise |
| `depth_estimation_failed` | MiDaS was unavailable; compositing ran without depth conditioning (flat result) |
| `fallback_model_used` | Primary model was unavailable; a lower-quality fallback was used |

---

### GET /jobs/{job_id}/trace
**Retrieve the full execution trace for a completed or failed job.**

Useful for debugging, evaluation, and demonstrating decision-making transparency to reviewers.

#### Example Request
```bash
curl http://localhost:8000/jobs/7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44/trace
```

#### Response — 200 OK
```json
{
  "success": true,
  "data": {
    "job_id": "7f3a2c91-4e1b-4d0a-b8f6-1c9e5a2d3b44",
    "total_duration_ms": 71400,
    "nodes": [
      {
        "node":         "input_guard",
        "status":       "success",
        "started_at":   "2026-06-03T10:20:00.000Z",
        "duration_ms":  42,
        "details": {
          "image_size_mb":     2.1,
          "image_resized":     false,
          "prompt_length":     76
        }
      },
      {
        "node":         "nlp_planner",
        "status":       "success",
        "started_at":   "2026-06-03T10:20:00.042Z",
        "duration_ms":  1850,
        "details": {
          "model":             "claude-sonnet-4-20250514",
          "confidence":        0.94,
          "required_nodes":    ["segmentation", "depth_estimation", "compositing"],
          "llm_tokens_used":   312
        }
      },
      {
        "node":         "segmentation",
        "status":       "success",
        "started_at":   "2026-06-03T10:20:01.892Z",
        "duration_ms":  18200,
        "details": {
          "model":             "replicate/segment-anything",
          "objects_masked":    ["person", "sky"],
          "mask_confidence":   0.71,
          "quality_flags":     []
        }
      },
      {
        "node":         "depth_estimation",
        "status":       "success",
        "started_at":   "2026-06-03T10:20:20.092Z",
        "duration_ms":  9300,
        "details": {
          "model":             "huggingface/midas",
          "output_format":     "16bit_grayscale_png"
        }
      },
      {
        "node":         "compositing",
        "status":       "success",
        "started_at":   "2026-06-03T10:20:29.392Z",
        "duration_ms":  42000,
        "details": {
          "model":             "replicate/sdxl-controlnet-depth",
          "generation_prompt": "dramatic sci-fi nebula sky, deep space, volumetric lighting",
          "depth_conditioned": true
        }
      }
    ]
  },
  "error": null
}
```

---

## Error Code Reference

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_INPUT` | 400 | Image format, size, or prompt failed validation |
| `JOB_NOT_FOUND` | 404 | No job exists for the given `job_id` |
| `PLAN_NOT_READY` | 409 | Plan endpoint accessed before planning is complete |
| `INVALID_STATE_TRANSITION` | 409 | Confirm called when job is not awaiting confirmation |
| `RESULT_NOT_READY` | 409 | Result endpoint accessed before job is done |
| `AMBIGUOUS_PROMPT` | 422 | Planner confidence below threshold; clarification required |
| `COST_GUARD_EXCEEDED` | 422 | Estimated job cost exceeds per-job spending limit |
| `SEGMENTATION_FAILED` | 500 | SAM API failed after retry |
| `DEPTH_ESTIMATION_FAILED` | 500 | MiDaS API failed after retry; partial result may still be available |
| `COMPOSITING_FAILED` | 500 | Stable Diffusion API failed after retry |
| `API_TIMEOUT` | 504 | Upstream model API did not respond within timeout window |

---

## Python SDK Example (Quick Start)

```python
import requests
import time

BASE_URL = "http://localhost:8000"

def run_vfx_job(image_path: str, prompt: str) -> dict:
    # 1. Submit job
    with open(image_path, "rb") as f:
        response = requests.post(f"{BASE_URL}/jobs", files={"image": f},
                                 data={"prompt": prompt})
    response.raise_for_status()
    job_id = response.json()["data"]["job_id"]
    print(f"Job submitted: {job_id}")

    # 2. Review the plan
    time.sleep(3)  # Give the planner a moment
    plan = requests.get(f"{BASE_URL}/jobs/{job_id}/plan").json()
    print("Agent plan:", plan["data"]["plan"]["plan_summary"])

    # 3. Poll until complete
    while True:
        status_resp = requests.get(f"{BASE_URL}/jobs/{job_id}").json()["data"]
        print(f"Status: {status_resp['status']} | Node: {status_resp.get('current_node')}")
        if status_resp["status"] in ("done", "failed"):
            break
        time.sleep(5)

    # 4. Retrieve result
    if status_resp["status"] == "done":
        result = requests.get(f"{BASE_URL}/jobs/{job_id}/result",
                              params={"format": "base64"}).json()
        if result["data"]["quality_flags"]:
            print("⚠ Quality flags:", result["data"]["quality_flags"])
        return result["data"]
    else:
        print("Job failed:", status_resp["errors"])
        return {}

# Usage
result = run_vfx_job("portrait.jpg", "Replace the sky with a sci-fi nebula")
```
