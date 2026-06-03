from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from app.routes.jobs import router as jobs_router
from app.routes.results import router as results_router


@asynccontextmanager
async def lifespan(app: FastAPI):
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
app.include_router(results_router)

