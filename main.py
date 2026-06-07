"""
ARC — Financial Distress Analyzer
FastAPI backend | LightGBM | SEC EDGAR | 35 features | Buyback-fix
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import os

from app.core.config import settings
from app.core.model_store import model_store
from app.routers import analyze, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading LightGBM model...")
    model_store.load()
    if model_store.is_loaded:
        print(f"Model loaded -- {len(model_store.feature_names)} features")
    else:
        print("No model found -- run train.py first, then restart")
    yield
    print("Shutdown.")


app = FastAPI(
    title="ARC Financial Distress Analyzer",
    description="LightGBM distress prediction with buyback detection & counter-strategy resistance.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(health.router,  prefix="/api/v1", tags=["Health"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard():
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return open(html_path, encoding="utf-8").read()
    return HTMLResponse("<h1>Place index.html in /static/</h1>", status_code=404)
