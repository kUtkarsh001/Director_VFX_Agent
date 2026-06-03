# 🎬 EskillVeda VFX Director Agent

> **AI-powered VFX orchestration** — describe what you want, Claude writes the plan, SAM segments the scene, MiDaS maps depth, and SDXL composites the final image. All in one REST API.

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-orange)](https://github.com/langchain-ai/langgraph)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Architecture

```
POST /jobs
  │
  ├─ input_guard        → validates image (≤10 MB, JPEG/PNG/WebP), prompt (≤500 chars)
  ├─ nlp_planner        → Claude claude-sonnet-4-20250514 (vision) → JSON execution plan
  │     └─ cost_guard   → aborts if projected spend > $0.02
  ├─ segmentation       → meta/sam-2 → per-object alpha masks
  ├─ depth_estimation   → cjwbw/midas → 16-bit depth map (non-fatal)
  ├─ compositing        → lucataco/sdxl-controlnet-depth → final PNG
  ├─ filter             → Pillow filters (greyscale, contrast, warm/cool) — no API cost
  └─ error_recovery     → inspects last error, retries once if transient
```

State is carried through a `VFXJobState` TypedDict. Routing is driven by `extracted_intent.required_nodes` — only the nodes specified in the plan are executed.

---

## Quickstart

### 1 — Prerequisites

| Tool | Version |
|---|---|
| Python | 3.12 |
| pip | any |

### 2 — Clone & install

```bash
git clone https://github.com/kUtkarsh001/Director_VFX_Agent.git
cd Director_VFX_Agent
python -m pip install -r requirements.txt
python -m pip install "fastapi==0.111.0" "uvicorn[standard]==0.29.0" numpy
```

### 3 — Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   ANTHROPIC_API_KEY=sk-ant-...
#   REPLICATE_API_TOKEN=r8_...
```

### 4 — Run

```bash
uvicorn app.main:app --reload --port 8000
```

API docs live at **http://localhost:8000/docs**

---

## Docker

```bash
# Build and start
docker compose up --build

# Stop
docker compose down
```

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/jobs` | Submit a new VFX job (multipart: `image` + `prompt`) |
| `GET` | `/jobs/{id}` | Poll job status |
| `GET` | `/jobs/{id}/plan` | View Claude's execution plan |
| `GET` | `/jobs/{id}/result` | Download final image (PNG) |
| `GET` | `/jobs/{id}/trace` | Per-node timing breakdown |
| `POST` | `/jobs/{id}/confirm` | Resume a paused job (`{"proceed": true}`) |

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "image=@photo.jpg" \
  -F "prompt=Replace the sky with a dramatic sunset" \
  -F "confirm_plan=false"
```

**Response 202**
```json
{
  "success": true,
  "data": {
    "job_id": "7f3a2c91-4b20-...",
    "status": "queued",
    "plan_url":   "http://localhost:8000/jobs/7f3a2c91.../plan",
    "status_url": "http://localhost:8000/jobs/7f3a2c91..."
  }
}
```

### Poll status

```bash
curl http://localhost:8000/jobs/7f3a2c91-...
```

---

## Project Structure

```
Director_VFX_Agent/
├── app/
│   ├── agent/
│   │   ├── graph.py          # LangGraph graph + routing functions
│   │   ├── state.py          # VFXJobState TypedDict
│   │   ├── timing.py         # start_timer / elapsed_ms helpers
│   │   └── nodes/
│   │       ├── input_guard.py
│   │       ├── nlp_planner.py
│   │       ├── segmentation.py
│   │       ├── depth_estimation.py
│   │       ├── compositing.py
│   │       ├── filter.py
│   │       └── error_recovery.py
│   ├── routes/
│   │   ├── jobs.py           # POST /jobs, GET /jobs/{id}
│   │   └── results.py        # GET /plan, GET /result, GET /trace, POST /confirm
│   ├── services/
│   │   ├── anthropic_client.py
│   │   ├── replicate_client.py
│   │   └── cost_guard.py
│   ├── storage/
│   │   └── job_store.py      # In-memory store (swap for Redis in prod)
│   └── main.py
├── tests/
│   ├── test_nodes.py         # 20 unit + mock-integration tests
│   └── test_routing.py
├── traces/                   # Output images written here
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Claude API key (`sk-ant-...`) |
| `REPLICATE_API_TOKEN` | ✅ | Replicate token (`r8_...`) |

---

## Running Tests

```bash
python -m pytest tests/test_nodes.py -v
# 20 tests — all mocked, no live API calls
```

---

## Cost Model

Each job is budgeted at **$0.02 maximum**. The cost guard aborts before any Replicate call if the estimate exceeds this limit.

| Node | Approx cost |
|---|---|
| nlp_planner (Claude vision) | $0.0060 |
| segmentation (SAM 2) | $0.0012 |
| depth_estimation (MiDaS) | $0.0005 |
| compositing (SDXL ControlNet) | $0.0025 |
| filter (Pillow) | $0.0000 |

---

## Roadmap

- [ ] Swap in-memory JobStore for Redis persistence
- [ ] Add job queue (Celery or RQ) for horizontal scaling
- [ ] Stream partial results via Server-Sent Events
- [ ] Add image output CDN upload (S3 / Cloudflare R2)
- [ ] Per-user rate limiting

---

## License

MIT © 2026 kUtkarsh001