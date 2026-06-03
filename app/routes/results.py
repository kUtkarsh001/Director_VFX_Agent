import base64
import io
import os
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, Response

from app.storage.job_store import job_store

router = APIRouter()


def _ok(data):
    return {"success": True, "data": data, "error": None}


def _err(code, msg, job_id=None):
    return {"success": False, "data": None,
            "error": {"code": code, "message": msg, "job_id": job_id}}


async def _get_or_404(job_id: str):
    record = await job_store.get_job(job_id)
    if record is None:
        return None, JSONResponse(
            404, _err("JOB_NOT_FOUND", f"No job: {job_id}", job_id)
        )
    return record, None


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/result — return final image bytes as PNG
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/result")
async def get_result(job_id: str):
    record, err = await _get_or_404(job_id)
    if err:
        return err
    state = record["state"]
    if state.get("status") != "done":
        return JSONResponse(409, _err("NOT_DONE", "Job is not yet complete.", job_id))
    img_bytes = state.get("final_image")
    if not img_bytes:
        return JSONResponse(404, _err("NO_RESULT", "Result image not found.", job_id))
    fmt = state.get("image_format", "png")
    return Response(content=img_bytes, media_type=f"image/{fmt}")


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/plan — return the NLP plan as JSON
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/plan")
async def get_plan(job_id: str):
    record, err = await _get_or_404(job_id)
    if err:
        return err
    state = record["state"]
    if not state.get("extracted_intent"):
        return JSONResponse(404, _err("NO_PLAN", "Plan not yet available.", job_id))
    return _ok({
        "job_id":           job_id,
        "execution_plan":   state.get("execution_plan", ""),
        "plan_confidence":  state.get("plan_confidence", 0.0),
        "extracted_intent": state.get("extracted_intent", {}),
    })


# ---------------------------------------------------------------------------
# GET /jobs/{job_id}/trace — return per-node timing breakdown
# ---------------------------------------------------------------------------
@router.get("/jobs/{job_id}/trace")
async def get_trace(job_id: str):
    record, err = await _get_or_404(job_id)
    if err:
        return err
    state = record["state"]
    return _ok({
        "job_id":        job_id,
        "nodes_executed": state.get("nodes_executed", []),
        "node_timings":  state.get("node_timings", {}),
        "errors":        state.get("errors", []),
        "quality_flags": state.get("quality_flags", []),
    })


# ---------------------------------------------------------------------------
# POST /jobs/{job_id}/confirm — resume execution after awaiting_confirmation
# ---------------------------------------------------------------------------

async def _safe_resume(job_id: str) -> None:
    """Background task: resumes graph execution from the stored planned state.
    Wraps errors so a crash here is written back to the job store instead of
    being silently swallowed (which would leave the job stuck at 'running').
    """
    import asyncio
    from app.agent.graph import build_vfx_graph

    try:
        graph = build_vfx_graph()
        record = await job_store.get_job(job_id)
        if record is None:
            return
        fresh  = dict(record["state"])
        result = await asyncio.to_thread(graph.invoke, fresh)
        await job_store.update_job(job_id, dict(result))
    except Exception as exc:
        record = await job_store.get_job(job_id)
        prior_errors = record["state"].get("errors", []) if record else []
        await job_store.update_job(job_id, {
            "status": "failed",
            "errors": prior_errors + [{
                "node":       "confirm",
                "error_code": "RESUME_FAILED",
                "message":    str(exc),
                "timestamp":  datetime.now(timezone.utc).isoformat(),
            }],
        })


@router.post("/jobs/{job_id}/confirm")
async def confirm_job(job_id: str, request: Request, background_tasks: BackgroundTasks):
    body    = await request.json() if request.headers.get("content-type") == \
        "application/json" else {}
    proceed = body.get("proceed", True)

    record, err = await _get_or_404(job_id)
    if err:
        return err
    state  = record["state"]
    status = state.get("status")

    if status != "awaiting_confirmation":
        return JSONResponse(409, _err("WRONG_STATE",
            f"Job is in '{status}', not 'awaiting_confirmation'.", job_id))

    if not proceed:
        await job_store.update_job(job_id, {"status": "failed", "errors":
            state.get("errors", []) + [{"node": "confirm", "error_code": "USER_CANCELLED",
            "message": "User cancelled.",
            "timestamp": datetime.now(timezone.utc).isoformat()}]})
        return _ok({"job_id": job_id, "status": "failed"})

    # Re-run from first tool node using the stored planned state.
    # Use FastAPI BackgroundTasks (same pattern as POST /jobs) so the server
    # can respond immediately while execution continues in the background.
    # Any crash in _safe_resume is written back to the job store — the job
    # will never be silently left stuck in 'running'.
    await job_store.update_job(job_id, {"status": "running"})
    background_tasks.add_task(_safe_resume, job_id)
    return _ok({"job_id": job_id, "status": "running"})
