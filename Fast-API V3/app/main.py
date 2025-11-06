from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes.job_routes import router as job_router
from app.routes.jd_agent_routes import router as agent_router

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import logging

from dotenv import load_dotenv


load_dotenv()  # This should be at the very top of your main.py

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Recruitment Ranking API",
    description="API d'analyse et de classement de CVs pour les RH",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Job Analysis",
            "description": "Endpoints pour l'analyse des offres d'emploi"
        }
    ]
)

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    if exc.status_code == 500:
        logger.error(f"500 Error: {exc.detail}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred. Please try again later.",
                "error_code": "INTERNAL_SERVER_ERROR",
                "detail": "Contact support with request ID: REQUEST_ID"  # Optional for production
            }
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logger.exception("Unhandled exception occurred")
    return JSONResponse(
        status_code=500,
        content={
            "message": "Something went wrong. Our team has been notified.",
            "error_code": "INTERNAL_SERVER_ERROR",
            "detail": "Contact support with request ID: REQUEST_ID"  # Hide in production
        }
    )

# Configuration CORS pour le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À ajuster en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Inclusion des routes
app.include_router(job_router, prefix="/api/v1")
app.include_router(agent_router)

# app.include_router(ranking_router, prefix="/api/v1")


@app.get("/health", tags=["Health Check"])
async def health_check():
    """Endpoint de vérification de l'état de l'API"""
    return {"status": "OK", "version": app.version}