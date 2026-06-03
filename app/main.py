from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routes.jobs import router as jobs_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure traces directory exists for result file serving
    os.makedirs("traces", exist_ok=True)
    yield


app = FastAPI(
    title="EskillVeda VFX Director Agent",
    description="AI-powered VFX orchestration: LangGraph + Claude + SAM + MiDaS + SD-XL",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs_router)

# Results router and static files for image serving added in Milestone 11
