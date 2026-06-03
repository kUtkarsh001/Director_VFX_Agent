import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Request, UploadFile, Form
from fastapi.responses import JSONResponse

from app.agent.graph import build_vfx_graph
from app.agent.nodes.input_guard import input_guard_node
from app.agent.state import VFXJobState
from app.storage.job_store import job_store

router = APIRouter()
_graph = build_vfx_graph()


def _ok(data: dict) -> dict:
    return {"success": True, "data": data, "error": None}


def _err(code: str, msg: str, node: str = None, job_id: str = None) -> dict:
    return {"success": False, "data": None,
            "error": {"code": code, "message": msg, "node": node, "job_id": job_id}}


async def _run_graph(job_id: str) -> None:
    """Background task: runs the full LangGraph pipeline for a job."""
    record = await job_store.get_job(job_id)
    if record is None:
        return
    state = dict(record["state"])
    confirm = record["confirm_plan"]

    await job_store.update_job(job_id, {"status": "planning"})
    try:
        result: VFXJobState = await asyncio.to_thread(_graph.invoke, state)
        # If confirm_plan=True and planner succeeded, pause before tool nodes
        if confirm and result.get("status") == "running":
            result["status"] = "awaiting_confirmation"
        await job_store.update_job(job_id, dict(result))
    except Exception as exc:
        await job_store.update_job(job_id, {
            "status": "failed",
            "errors": state.get("errors", []) + [{
                "node": "graph_runner", "error_code": "GRAPH_EXECUTION_ERROR",
                "message": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }],
        })


@router.post("/jobs", status_code=202)
async def create_job(
    request:          Request,
    background_tasks: BackgroundTasks,
    image:            UploadFile,
    prompt:           str  = Form(...),
    response_format:  str  = Form("url"),
    confirm_plan:     bool = Form(False),
):
    image_bytes = await image.read()
    job_id      = str(uuid.uuid4())
    base_url    = str(request.base_url).rstrip("/")

    initial_state: VFXJobState = {
        "job_id": job_id, "original_image": image_bytes,
        "image_format": "", "user_prompt": prompt,
        "extracted_intent": {}, "execution_plan": "",
        "plan_confidence": 0.0, "masks": None,
        "mask_confidence": None, "depth_map": None, "final_image": None,
        "nodes_executed": [], "errors": [], "quality_flags": [],
        "status": "queued", "retry_count": 0, "retry_target": None,
        "node_timings": {},
    }

    # Fast inline validation — catches bad input before any background cost
    validated = input_guard_node(dict(initial_state))
    if validated["status"] == "failed":
        err = validated["errors"][0]
        return JSONResponse(400, _err(err["error_code"], err["message"], "input_guard"))

    await job_store.create_job(job_id, initial_state, confirm_plan=confirm_plan)
    background_tasks.add_task(_run_graph, job_id)

    return _ok({
        "job_id":     job_id, "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "plan_url":   f"{base_url}/jobs/{job_id}/plan",
        "status_url": f"{base_url}/jobs/{job_id}",
    })


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    record = await job_store.get_job(job_id)
    if record is None:
        return JSONResponse(404, _err("JOB_NOT_FOUND", f"No job: {job_id}", job_id=job_id))

    state    = record["state"]
    base_url = str(request.base_url).rstrip("/")
    status   = state["status"]

    data = {
        "job_id":          job_id,   "status":          status,
        "current_node":    None,
        "nodes_completed": state.get("nodes_executed", []),
        "created_at":      record["created_at"],
        "updated_at":      record["updated_at"],
        "quality_flags":   state.get("quality_flags", []),
        "errors":          state.get("errors", []),
    }
    if status == "done":
        data["result_url"] = f"{base_url}/jobs/{job_id}/result"
        data["trace_url"]  = f"{base_url}/jobs/{job_id}/trace"

    return _ok(data)
