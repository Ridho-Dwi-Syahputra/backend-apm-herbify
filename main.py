# =============================================================================
# main.py — FastAPI Entry Point (Hanya MobileNetV2)
# =============================================================================
# Jalankan: python -m uvicorn main:app --reload --port 8000
# Akses: http://localhost:8000
# Docs: http://localhost:8000/docs (Swagger UI)
# =============================================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.prediction_service import prediction_service


@asynccontextmanager
async def lifespan(app):
    """Load model MobileNetV2 saat server startup."""
    print("=" * 50)
    print("Herbify Backend API Server")
    print("=" * 50)

    prediction_service.load_model()

    print("\nServer ready!")
    print("API Docs: http://localhost:8000/docs")
    print("=" * 50)

    yield

    print("Server shutting down...")


from fastapi.responses import FileResponse
from pathlib import Path

app = FastAPI(
    title="Herbify API",
    description="API klasifikasi tanaman obat menggunakan MobileNetV2",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    icon_path = Path(r"D:\Local Disk D\Tugas\Semester 6\Aplikasi Pembelajaran Mesin\Project TB\frontend\herbify\assets\images\herbify_logo_icon.png")
    if icon_path.exists():
        return FileResponse(str(icon_path))
    return {"message": "No favicon"}

# CORS — izinkan Flutter app mengakses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {
        "app": "Herbify API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "predict": "POST /api/predict",
            "health": "GET /api/health",
            "plants": "GET /api/plants",
        },
    }
