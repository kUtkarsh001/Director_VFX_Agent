# Product Requirements Document
## EskillVeda VFX Director Agent
**Version:** 1.1 | **Author:** AI Engineer Candidate | **Date:** 2026-06-03

---

## 1. Executive Summary

The EskillVeda VFX Director Agent is an AI-powered orchestration system that accepts a plain-English description of a visual effects task and autonomously produces a composited output image. Rather than wrapping a single model, it acts as a reasoning "Director" — decomposing the user's intent, selecting and sequencing the right specialist models (segmentation, depth estimation, generative inpainting), and synthesising their outputs into a production-quality result.

The system is built for the AI Engineer Intern evaluation and is designed to demonstrate mastery of agentic AI pipelines, LLM-driven tool orchestration, and scalable modular architecture — not brute-force model training.

---

## 2. Problem Statement

Producing a single professional VFX composite (e.g., background removal + sky replacement + lighting match) currently requires a visual effects artist with access to multiple specialised tools and several hours of work. Hobbyists, educators, and content creators lack this expertise. Existing automated tools (e.g., one-click background removers) are rigid pipelines that cannot handle compound or creative prompts.

**The gap:** There is no conversational, instruction-following VFX tool that:
- Understands compound, ambiguous instructions in natural language.
- Selects and chains the *right* models based on intent.
- Produces outputs that respect scene geometry and lighting context.

---

## 3. Goals & Non-Goals

### 3.1 Goals
| # | Goal | Priority |
|---|------|----------|
| G1 | Accept an image and a free-text VFX prompt | Must Have |
| G2 | Parse and decompose the intent via an LLM planner | Must Have |
| G3 | Route the job through the correct subset of tools (segmentation, depth, compositing) | Must Have |
| G4 | Return a composited output image | Must Have |
| G5 | Return a human-readable execution plan *before* running, allowing confirmation | Should Have |
| G6 | Gracefully handle model API failures with fallback strategies | Should Have |
| G7 | Expose a simple REST API for programmatic use | Should Have |
| G8 | Log structured trace data for debugging and evaluation | Should Have |
| G9 | Support short video clips (≤10s) as input | Could Have (Stretch) |

### 3.2 Non-Goals
- Training or fine-tuning any vision model from scratch.
- Building a full web UI / frontend application.
- Real-time (≤1s) processing — batch latency of 30–120s is acceptable.
- Multi-user authentication or billing infrastructure.
- Support for audio or non-visual media.

---

## 4. User Stories

**Primary Persona:** A content creator or student exploring AI-assisted video production.

| ID | As a… | I want to… | So that… | Acceptance Criteria |
|----|--------|-----------|----------|---------------------|
| US-01 | Content creator | Submit an image + text prompt via API | I can automate VFX without coding | POST /jobs returns a job_id within 2s |
| US-02 | Content creator | See what the agent *plans to do* before it runs | I can catch misinterpretations early | GET /jobs/{id}/plan returns a structured plan before execution starts |
| US-03 | Content creator | Receive a composited image as output | I can use the result in my project | GET /jobs/{id}/result returns a downloadable image |
| US-04 | Developer | Query job status | I can build async polling into my app | GET /jobs/{id} returns status ∈ {queued, planning, running, done, failed} |
| US-05 | Developer | Receive structured error messages | I can debug failures | Error responses include node_name, error_code, and message |
| US-06 | Evaluator | Read a structured execution log | I can verify the agent made correct decisions | GET /jobs/{id}/trace returns node-by-node execution log |

---

## 5. Functional Requirements

### 5.1 Input Handling
- **FR-01:** Accept image uploads in JPEG, PNG, or WebP format (max 10 MB).
- **FR-02:** Accept a text prompt string (max 500 characters).
- **FR-03:** Validate input before submitting to the agent graph; return a descriptive error for invalid inputs.

### 5.2 NLP Planning
- **FR-04:** The LLM Planner must extract: `target_objects` (list), `actions` (list), and `required_nodes` (list) from the prompt.
- **FR-05:** The Planner must output a human-readable execution plan summarising its interpretation before dispatching to tool nodes.
- **FR-06:** If the prompt is ambiguous (confidence < threshold), the Planner must return a clarification request rather than proceeding blindly.

### 5.3 Tool Orchestration
- **FR-07:** The segmentation node must be triggered only if an object needs to be isolated or removed.
- **FR-08:** The depth estimation node must be triggered only if spatial placement or lighting matching is required.
- **FR-09:** The compositing node must receive masks and depth map as inputs when they are available.
- **FR-10:** Simple filter operations (e.g., greyscale, contrast) must be executable without invoking any external model API.

### 5.4 Error Handling
- **FR-11:** If a tool node fails, the agent must retry once with the same inputs before entering the `failed` state.
- **FR-12:** If the segmentation model returns a low-confidence mask (IoU < 0.6), the Planner must log a warning and proceed with a degraded-quality flag rather than silently failing.
- **FR-13:** All node-level errors must be written to the job's `errors` state field.

### 5.5 Output
- **FR-14:** The output image must match the input image's original resolution.
- **FR-15:** The API must provide a direct download URL valid for at least 1 hour.
- **FR-16:** A structured execution trace must be stored and retrievable after job completion.

---

## 6. Non-Functional Requirements

| ID | Category | Requirement |
|----|----------|-------------|
| NFR-01 | Latency | End-to-end job completion ≤ 90s for a 1080p image under normal API conditions |
| NFR-02 | Reliability | The API server must remain responsive even when a downstream model API is unavailable |
| NFR-03 | Maintainability | Each tool node must be independently replaceable without modifying other nodes |
| NFR-04 | Observability | All LLM calls and tool invocations must emit structured log events |
| NFR-05 | Security | User-uploaded images must not be persisted beyond 24 hours |
| NFR-06 | Portability | The system must run locally via Docker Compose with environment variables replacing API keys |

---

## 7. Constraints

- **Time:** 24-hour development window. All scope decisions must respect this hard limit.
- **Compute:** Heavy vision models (SAM, MiDaS, Stable Diffusion) must be called via hosted API (HuggingFace Inference API or Replicate) rather than run locally.
- **Budget:** External API calls must be managed; a cost-guard mechanism should prevent runaway spending on a single malformed job.
- **Evaluation Context:** The project will be assessed by technical evaluators. The architecture must be legible, defensible, and demonstrate conceptual correctness — not just functional output.

---

## 8. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Prompt interpretation accuracy | ≥ 85% correct tool routing on 10 test prompts | Manual evaluation by reviewer |
| End-to-end job success rate | ≥ 80% on valid inputs | Automated test suite |
| P95 job latency | ≤ 90 seconds | Job completion timestamps |
| Plan preview correctness | Reviewer agrees with plan in ≥ 8/10 cases | Manual review |
| Code modularity | Each node < 80 lines of code | Code review |

---

## 9. Delivery Phases

Given the 24-hour constraint, scope is split into a mandatory core and an optional stretch layer.

### Phase 1 — Core (Hours 0–18)
- LangGraph state machine with all four primary nodes.
- Claude-powered NLP Planner with plan preview output.
- SAM integration (segmentation), MiDaS integration (depth), Stable Diffusion + ControlNet (compositing).
- FastAPI server with `/jobs` POST and GET endpoints.
- Basic structured logging.

### Phase 2 — Hardening (Hours 18–22)
- Retry logic and error recovery node.
- Low-confidence mask warning system.
- Cost-guard on API calls.
- `/jobs/{id}/plan` and `/jobs/{id}/trace` endpoints.

### Phase 3 — Stretch (Hours 22–24)
- Simple greyscale / contrast filter path (no external API).
- Docker Compose packaging.
- README with architecture diagram and example curl commands.

---

## 10. Open Questions

| # | Question | Impact | Resolution Path |
|---|----------|--------|----------------|
| OQ-01 | Does Replicate's ControlNet API expose a depth-conditioned endpoint? | High — affects compositing fidelity | Test API during Phase 1 |
| OQ-02 | What is SAM's API latency on Replicate for a 1080p image? | Medium — affects overall job latency SLA | Benchmark during Phase 1 |
| OQ-03 | Is LangSmith tracing available under the free tier for this evaluation? | Low — affects observability richness | Check docs; fallback to local logging |
