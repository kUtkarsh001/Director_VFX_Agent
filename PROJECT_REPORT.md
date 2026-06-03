# EskillVeda VFX Director Agent
**Project Report**

---

## 1. Executive Summary
The **EskillVeda VFX Director Agent** is an intelligent, autonomous visual effects orchestration pipeline. The project aims to simplify complex image manipulation tasks by allowing users to provide natural language prompts along with an input image. The system uses an AI-powered agent to parse the intent, formulate an execution plan, and chain together state-of-the-art computer vision and generative models to produce the desired visual output.

By abstracting away the manual labor of segmentation, depth mapping, and compositing, this agent serves as a fully automated "VFX Director."

---

## 2. Introduction
Traditional visual effects and image manipulation workflows require specialized software, manual masking, and deep technical expertise. The goal of this project is to democratize high-end image editing by leveraging recent advancements in Large Language Models (LLMs) and Generative AI.

The application provides a RESTful API where users can submit raw images and plain English instructions (e.g., "Replace the sky with a dramatic sunset"). The backend orchestration engine intelligently decides which AI models are required, routes the image through those models sequentially, and returns a photorealistic composite image.

---

## 3. System Architecture & Workflow
The system is designed as a state machine using a directed graph. The architecture ensures modularity, robust error handling, and separation of concerns.

### The Pipeline Workflow:
1. **Input Guard**: Validates incoming images (size ≤10 MB, JPEG/PNG/WebP format) and text prompts (≤500 chars). Automatically resizes images exceeding 2048px. Runs inline on the request thread for fast 400 responses, and is skipped in the background graph to avoid double processing.
2. **NLP Planner (The Brain)**: Uses Claude 3.5 Sonnet to analyze the image and prompt. It outputs a structured JSON plan detailing the target objects, required actions, and the specific AI nodes needed. Records full timing data for the `/trace` endpoint.
3. **Cost Guard**: A pre-flight check integrated into the NLP planner that estimates the API cost of the execution plan. It aborts the job if the projected cost exceeds the configurable budget (default $0.02, set via `MAX_COST_PER_JOB_USD` environment variable).
4. **Segmentation Node**: Uses **Grounded-SAM** (Grounding DINO + SAM) to generate precise alpha masks for the target objects identified by the planner. Unlike raw SAM-2 which requires spatial coordinates, Grounded-SAM accepts free-text object labels — matching our text-driven pipeline perfectly.
5. **Depth Estimation Node**: Uses **MiDaS** to generate a 16-bit monocular depth map of the scene, allowing generative models to understand the 3D geometry of the image. Failure is non-fatal: the pipeline flags it and continues.
6. **Compositing Node**: Uses **Stable Diffusion XL (SDXL) ControlNet (Depth)** to generate the final image. Before generation, it performs a **perimeter-based lighting analysis** that samples border pixels to detect the scene's ambient color temperature (warm/cool/bright/dim/neutral) and injects the result into the SDXL prompt for color-matched compositing. It seamlessly blends the prompt, the original image, the alpha mask, and the depth map.
7. **Filter Node**: A zero-cost node utilizing Pillow and NumPy to apply local image processing: greyscale, sharpen, blur, contrast, brightness, warm/cool tints.
8. **Error Recovery**: If an external API fails, this node inspects the latest error code and can intelligently retry the specific failing node once. Routing functions check only the *latest* error to prevent infinite loops after successful retries.

---

## 4. Technology Stack
| Component | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.12 | AI/ML ecosystem, async support, type hints |
| **Web Framework** | FastAPI | High-performance async REST, auto-generated OpenAPI docs |
| **Orchestration** | LangGraph | Graph-based state machine with conditional routing and loops |
| **LLM Provider** | Anthropic API (Claude 3.5 Sonnet) | Vision + reasoning for structured JSON plan extraction |
| **Segmentation** | Grounded-SAM (via Replicate) | Text-prompted object detection + segmentation |
| **Depth Mapping** | MiDaS (via Replicate) | Monocular depth estimation for 3D scene understanding |
| **Compositing** | SDXL ControlNet Depth (via Replicate) | Depth-conditioned generative image editing |
| **Image Processing** | Pillow + NumPy | Local filters, mask union, lighting analysis (zero API cost) |
| **Deployment** | Docker & Docker Compose | Multi-stage builds, non-root runtime, health checks |

---

## 5. Key Features
* **Asynchronous Execution**: Jobs are processed in the background, allowing the FastAPI server to handle high concurrency without blocking. Clients receive a polling URL to track progress.
* **Graceful Degradation**: If non-critical nodes (like depth estimation) fail, the pipeline flags the issue but proceeds to compositing using standard inpainting, ensuring the user still receives an output.
* **Lighting Matching**: The compositing node analyzes the original image's border pixels to estimate ambient lighting conditions and injects scene-appropriate lighting descriptors into the SDXL prompt, ensuring generated content is color-matched to the original scene.
* **Deterministic Output**: The use of structured JSON schemas enforces strict formatting from the LLM, preventing pipeline crashes due to hallucinated or malformed text.
* **Traceability**: The API provides a `/trace` endpoint that breaks down the execution time and status of every individual node (including the NLP planner), allowing for deep profiling and debugging.
* **Configurable Budget**: The per-job cost limit is configurable via environment variable (`MAX_COST_PER_JOB_USD`), not hardcoded, following best practices for deployment flexibility.

---

## 6. Challenges & Solutions
* **Challenge**: LLMs often wrap JSON outputs in Markdown code blocks (e.g., ````json ... ````), which breaks Python's standard `json.loads()`.
  * **Solution**: Implemented a custom sanitization function in the Anthropic client to strip markdown fences before parsing.
* **Challenge**: Chaining multiple heavy generative models can easily lead to runaway costs.
  * **Solution**: Introduced a strict, pre-execution `Cost Guard` that calculates the projected cost based on the execution plan and aborts the job if it exceeds the configurable budget.
* **Challenge**: External APIs (like Replicate) can occasionally timeout or fail transiently.
  * **Solution**: Built an `error_recovery` node in LangGraph with loop-safe routing — each routing function checks only the *latest* error and verifies the `retry_target` flag, preventing infinite retry loops after successful recoveries.
* **Challenge**: SAM-2 requires spatial coordinates (point/box prompts), but our pipeline is entirely text-driven.
  * **Solution**: Switched to Grounded-SAM, which internally uses Grounding DINO for zero-shot text-based object detection before passing bounding boxes to SAM for precise mask generation.
* **Challenge**: Generated content must match the source image's lighting to look realistic.
  * **Solution**: Implemented perimeter pixel sampling to estimate ambient color temperature and brightness, injecting scene-specific lighting descriptors into the SDXL compositing prompt.

---

## 7. Testing
The project includes a comprehensive test suite of **37 tests** across two files:

| Test File | Tests | Coverage |
|---|---|---|
| `test_nodes.py` | 20 | Input validation, prompt checks, image resizing, Claude client mocking, fence stripping, planner JSON parsing, confidence/clarification routing |
| `test_routing.py` | 17 | All 5 routing functions: planner branching, segmentation loop prevention, depth non-fatal fallthrough, compositing terminal states, error recovery retry logic |

All tests are fully mocked — no live API calls are required. Run with:
```bash
python -m pytest tests/ -v
```

---

## 8. Conclusion
The EskillVeda VFX Director Agent successfully demonstrates how complex, multi-step creative workflows can be fully automated using a graph-based orchestration framework. By combining the reasoning capabilities of LLMs with the raw generative power of specialized vision models, the system delivers high-quality visual effects through a simple, intuitive API. The architecture is designed for production readiness with configurable budgets, loop-safe error recovery, lighting-aware compositing, and comprehensive test coverage.

---

## 9. Future Scope
While the current implementation is fully functional, there are several avenues for future expansion:
1. **Video Processing**: Extending the pipeline to process short video clips frame-by-frame, ensuring temporal consistency between frames using models like Ebsynth or AnimateDiff.
2. **Persistent Job Storage**: Swapping the current in-memory job store for a robust database like Redis or PostgreSQL to persist job states across server restarts.
3. **Queue Management**: Integrating Celery or RQ (Redis Queue) to manage heavy workloads and scale worker nodes horizontally across multiple servers. The Docker Compose file already includes a scaffolded worker service definition ready for activation.
4. **WebSocket / SSE Integration**: Replacing HTTP polling with Server-Sent Events (SSE) or WebSockets to stream real-time execution logs and progress to the frontend client.
5. **Human-in-the-Loop (HITL)**: Expanding the `awaiting_confirmation` state to allow users to visually inspect and manually adjust the generated masks before the final compositing step takes place.
6. **Cloud Storage Integration**: Automatically uploading the final high-resolution composite images to AWS S3 or Cloudflare R2 and returning a CDN link to the user.
7. **Advanced Lighting Analysis**: Enhancing the current perimeter-sampling approach with a dedicated light direction estimation model to produce more precise shadow and highlight matching.
