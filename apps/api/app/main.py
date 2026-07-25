from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes.reconstruction import router as reconstruction_router
from app.schemas import HealthResponse
from app.services.reconstruction import create_reconstructor


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.reconstructor = create_reconstructor(settings.model_path)
    yield


app = FastAPI(
    title="LISS-IV Reconstruction API",
    version="0.1.0",
    description=(
        "Cloud detection, mask-guided multispectral reconstruction, "
        "and uncertainty analysis for LISS-IV satellite imagery."
    ),
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.include_router(reconstruction_router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    return {
        "service": "LISS-IV Reconstruction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    reconstructor = app.state.reconstructor
    return HealthResponse(
        status="ok",
        service="liss-iv-reconstruction",
        model_ready=True,
        inference_mode=reconstructor.mode,
    )
