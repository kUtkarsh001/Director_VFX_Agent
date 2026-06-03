import asyncio
from datetime import datetime, timezone
from typing import Optional

from app.agent.state import VFXJobState


class JobStore:
    """In-memory job store backed by a plain dict, guarded by asyncio.Lock.
    Interface is deliberately thin so it can be swapped for Redis without
    touching any agent or route code."""

    def __init__(self):
        self._store: dict[str, dict] = {}
        self._lock  = asyncio.Lock()

    async def create_job(
        self, job_id: str, initial_state: VFXJobState, *, confirm_plan: bool = False
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        async with self._lock:
            self._store[job_id] = {
                "state":        dict(initial_state),
                "created_at":   now,
                "updated_at":   now,
                "confirm_plan": confirm_plan,
            }

    async def get_job(self, job_id: str) -> Optional[dict]:
        async with self._lock:
            return self._store.get(job_id)

    async def update_job(self, job_id: str, partial_state: dict) -> None:
        async with self._lock:
            if job_id not in self._store:
                return
            self._store[job_id]["state"].update(partial_state)
            self._store[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    async def list_jobs(self) -> list[str]:
        async with self._lock:
            return list(self._store.keys())


# Module-level singleton — imported by routes and background runners
job_store = JobStore()
