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
1. **Input Guard**: Validates incoming images (size, format) and text prompts.
2. **NLP Planner (The Brain)**: Uses Claude 3.5 Sonnet to analyze the image and prompt. It outputs a structured JSON plan detailing the target objects, required actions, and the specific AI nodes needed.
3. **Cost Guard**: A pre-flight check that estimates the API cost of the execution plan. It aborts the job if the projected cost exceeds the strict $0.02 budget.
4. **Segmentation Node**: Uses Meta's **SAM-2 (Segment Anything 2)** to generate precise alpha masks for the target objects identified by the planner.
5. **Depth Estimation Node**: Uses **MiDaS** to generate a 16-bit monocular depth map of the scene, allowing generative models to understand the 3D geometry of the image.
6. **Compositing Node**: Uses **Stable Diffusion XL (SDXL) ControlNet (Depth)** to generate the final image. It seamlessly blends the prompt, the original image, the alpha mask, and the depth map.
7. **Filter Node**: A zero-cost fallback node utilizing local Python libraries to apply basic color correction, contrast, and tints.
8. **Error Recovery**: If an external API fails, this node can catch the error and intelligently retry the failing step before marking the job as failed.

---

## 4. Technology Stack
* **Language**: Python 3.12
* **Web Framework**: FastAPI (for high-performance, asynchronous REST endpoints)
* **Orchestration**: LangGraph (for graph-based state management and conditional routing)
* **LLM Provider**: Anthropic API (Claude 3.5 Sonnet for vision and reasoning)
* **Model Inference**: Replicate API (Serverless GPU provider for SAM-2, MiDaS, and SDXL)
* **Image Processing**: Pillow (PIL)
* **Deployment**: Docker & Docker Compose (Multi-stage builds for lean containerization)

---

## 5. Key Features
* **Asynchronous Execution**: Jobs are processed in the background, allowing the FastAPI server to handle high concurrency without blocking. Clients receive a polling URL to track progress.
* **Graceful Degradation**: If non-critical nodes (like depth estimation) fail, the pipeline flags the issue but proceeds to compositing using standard inpainting, ensuring the user still receives an output.
* **Deterministic Output**: The use of structured JSON schemas enforces strict formatting from the LLM, preventing pipeline crashes due to hallucinated or malformed text.
* **Traceability**: The API provides a `/trace` endpoint that breaks down the execution time and status of every individual node, allowing for deep profiling and debugging.

---

## 6. Challenges & Solutions
* **Challenge**: LLMs often wrap JSON outputs in Markdown code blocks (e.g., ````json ... ````), which breaks Python's standard `json.loads()`.
  * **Solution**: Implemented a custom sanitization function in the Anthropic client to strip markdown fences before parsing.
* **Challenge**: Chaining multiple heavy generative models can easily lead to runaway costs.
  * **Solution**: Introduced a strict, pre-execution `Cost Guard` node that calculates the projected cost based on the execution plan and aborts the job if it exceeds $0.02.
* **Challenge**: External APIs (like Replicate) can occasionally timeout or fail transiently.
  * **Solution**: Built an `error_recovery` node in LangGraph that allows for automatic, limited retries of specific failing nodes.

---

## 7. Conclusion
The EskillVeda VFX Director Agent successfully demonstrates how complex, multi-step creative workflows can be fully automated using a graph-based orchestration framework. By combining the reasoning capabilities of LLMs with the raw generative power of specialized vision models, the system delivers high-quality visual effects through a simple, intuitive API.

---

## 8. Future Scope
While the current implementation is fully functional, there are several avenues for future expansion:
1. **Video Processing**: Extending the pipeline to process short video clips frame-by-frame, ensuring temporal consistency between frames using models like Ebsynth or AnimateDiff.
2. **Persistent Job Storage**: Swapping the current in-memory job store for a robust database like Redis or PostgreSQL to persist job states across server restarts.
3. **Queue Management**: Integrating Celery or RQ (Redis Queue) to manage heavy workloads and scale worker nodes horizontally across multiple servers.
4. **WebSocket / SSE Integration**: Replacing HTTP polling with Server-Sent Events (SSE) or WebSockets to stream real-time execution logs and progress to the frontend client.
5. **Human-in-the-Loop (HITL)**: Expanding the `awaiting_confirmation` state to allow users to visually inspect and manually adjust the generated masks before the final compositing step takes place.
6. **Cloud Storage Integration**: Automatically uploading the final high-resolution composite images to AWS S3 or Cloudflare R2 and returning a CDN link to the user.
