"""
FastAPI Movie Recommendation System
Main application entry point with CORS, middleware, and routing configuration.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
from contextlib import asynccontextmanager

from app.api import movies, recommendations
from app.core.config import settings
from app.core.ml_models import ModelManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting Movie Recommendation System API...")
    
    # Initialize ML models on startup
    try:
        model_manager = ModelManager()
        await model_manager.load_models()
        app.state.model_manager = model_manager
        logger.info("ML models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ML models: {e}")
        # Continue without models for now - will use fallbacks
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Movie Recommendation System API...")


# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation System API",
    description="A modern movie recommendation system using collaborative filtering, content-based filtering, and hybrid approaches",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests with timing information."""
    start_time = time.time()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response with timing
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")
    
    # Add timing header
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with proper logging and response."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "timestamp": time.time()
        }
    )


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring and deployment verification."""
    return {
        "status": "healthy",
        "service": "Movie Recommendation System API",
        "version": "1.0.0",
        "timestamp": time.time()
    }


# Include API routers
app.include_router(movies.router, prefix="/api", tags=["movies"])
app.include_router(recommendations.router, prefix="/api", tags=["recommendations"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Movie Recommendation System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health"
    }