"""
Configuration settings for the Movie Recommendation System API.
"""

from pydantic import Field, ConfigDict
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    API_TITLE: str = "Movie Recommendation System API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "A modern movie recommendation system using ML algorithms"
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:3000",  # React development server
            "http://localhost:5173",  # Vite development server
            "https://movie-recommender-frontend.vercel.app",  # Production frontend
        ],
        description="Allowed CORS origins"
    )
    
    # Data Configuration
    DATA_DIR: str = Field(default="data", description="Directory containing data files")
    MODELS_DIR: str = Field(default="data/models", description="Directory containing ML models")
    
    # TMDB API Configuration
    TMDB_API_KEY: Optional[str] = Field(default=None, description="TMDB API key for movie posters")
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_IMAGE_BASE_URL: str = "https://image.tmdb.org/t/p/w500"
    
    # ML Model Configuration
    DEFAULT_ALGORITHM: str = Field(default="hybrid", description="Default recommendation algorithm")
    DEFAULT_NUM_RECOMMENDATIONS: int = Field(default=20, description="Default number of recommendations")
    MIN_RATINGS_REQUIRED: int = Field(default=15, description="Minimum ratings required for recommendations")
    
    # Performance Configuration
    ENABLE_CACHING: bool = Field(default=True, description="Enable response caching")
    CACHE_TTL: int = Field(default=3600, description="Cache TTL in seconds")
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, description="Maximum concurrent requests")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Development Configuration
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    RELOAD: bool = Field(default=False, description="Enable auto-reload in development")
    
    # Security Configuration
    ALLOWED_HOSTS: List[str] = Field(default=["*"], description="Allowed host headers")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        json_schema_extra={
            "example": {
                "TMDB_API_KEY": "your_tmdb_api_key_here",
                "DEBUG": "false",
                "LOG_LEVEL": "INFO",
                "CORS_ORIGINS": '["http://localhost:3000", "https://yourdomain.com"]'
            }
        }
    )


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency to get settings instance."""
    return settings