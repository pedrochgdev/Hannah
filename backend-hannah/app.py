"""
app.py
------
FastAPI entry point.
Sirve el pipeline de IA Y el frontend estatico en el mismo puerto (8000).
Los usuarios de la LAN abren: http://IP-de-la-maquina:8000

Start:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router
from config import settings

app = FastAPI(
    title=settings.app_title,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS abierto — el sistema no es publico, solo LAN interna
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

# Servir archivos estaticos del frontend (CSS, JS)
_FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend-hannah")
_STATIC   = os.path.join(_FRONTEND, "static")

if os.path.isdir(_STATIC):
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")


@app.get("/")
def serve_index():
    """Sirve index.html. Los usuarios abren http://IP:8000 y ven el chat."""
    return FileResponse(os.path.join(_FRONTEND, "templates", "index.html"))
