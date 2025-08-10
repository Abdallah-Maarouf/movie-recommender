"""
Recommendation models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Dict, List, Optional, Literal
from datetime import datetime

from .movie import MovieSummary


class RatingRequest(BaseModel):
    """Request model for user ratings input."""
    ratings: Dict[int, float] = Field(
        ..., 
        description="Dictionary mapping movie IDs to rating values (1.0-5.0)"
    )
    algorithm: Optional[Literal["collaborative", "content", "hybrid"]] = Field(
        "hybrid",
        description="Recommendation algorithm to use"
    )
    num_recommendations: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    
    @field_validator('ratings')
    @classmethod
    def validate_ratings(cls, v):
        """Validate rating values and ensure minimum requirements."""
        if not v:
            raise ValueError("At least one rating is required")
        
        for movie_id, rating in v.items():
            if not isinstance(movie_id, int) or movie_id <= 0:
                raise ValueError(f"Invalid movie ID: {movie_id}")
            if not isinstance(rating, (int, float)) or not (1.0 <= rating <= 5.0):
                raise ValueError(f"Rating must be between 1.0 and 5.0, got: {rating}")
        
        if len(v) < 15:
            raise ValueError("At least 15 ratings are required for recommendations")
        
        return v
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "ratings": {
                    "1": 4.0,
                    "2": 3.5,
                    "3": 5.0,
                    "4": 2.0,
                    "5": 4.5
                },
                "algorithm": "hybrid",
                "num_recommendations": 20
            }
        }
    )


class RecommendationItem(BaseModel):
    """Individual recommendation item with prediction details."""
    movie: MovieSummary = Field(..., description="Movie details")
    predicted_rating: float = Field(
        ..., 
        ge=1.0, 
        le=5.0, 
        description="Predicted rating for this movie"
    )
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Confidence score for this recommendation"
    )
    explanation: str = Field(..., description="Human-readable explanation for recommendation")
    algorithm_used: str = Field(..., description="Algorithm that generated this recommendation")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "movie": {
                    "id": 50,
                    "title": "The Usual Suspects",
                    "genres": ["Crime", "Thriller"],
                    "year": 1995,
                    "poster_url": "https://image.tmdb.org/t/p/w500/poster.jpg",
                    "average_rating": 4.2
                },
                "predicted_rating": 4.3,
                "confidence": 0.85,
                "explanation": "Because you liked Pulp Fiction and The Shawshank Redemption",
                "algorithm_used": "hybrid"
            }
        }
    )


class RecommendationResponse(BaseModel):
    """Complete recommendation response."""
    recommendations: List[RecommendationItem] = Field(
        ..., 
        description="List of recommended movies"
    )
    total_ratings: int = Field(..., description="Total number of ratings provided")
    algorithm_used: str = Field(..., description="Primary algorithm used")
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0, 
        description="Overall confidence in recommendations"
    )
    processing_time: float = Field(..., description="Time taken to generate recommendations (seconds)")
    metadata: Optional[Dict] = Field(None, description="Additional metadata about the recommendation process")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "recommendations": [
                    {
                        "movie": {
                            "id": 50,
                            "title": "The Usual Suspects",
                            "genres": ["Crime", "Thriller"],
                            "year": 1995,
                            "poster_url": "https://image.tmdb.org/t/p/w500/poster.jpg",
                            "average_rating": 4.2
                        },
                        "predicted_rating": 4.3,
                        "confidence": 0.85,
                        "explanation": "Because you liked Pulp Fiction and The Shawshank Redemption",
                        "algorithm_used": "hybrid"
                    }
                ],
                "total_ratings": 18,
                "algorithm_used": "hybrid",
                "confidence_score": 0.82,
                "processing_time": 0.245,
                "metadata": {
                    "collaborative_weight": 0.7,
                    "content_weight": 0.3,
                    "fallback_used": False
                }
            }
        }
    )


class UpdateRecommendationRequest(BaseModel):
    """Request model for updating recommendations with new ratings."""
    existing_ratings: Dict[int, float] = Field(..., description="Previously provided ratings")
    new_ratings: Dict[int, float] = Field(..., description="New ratings to incorporate")
    algorithm: Optional[Literal["collaborative", "content", "hybrid"]] = Field(
        "hybrid",
        description="Recommendation algorithm to use"
    )
    num_recommendations: Optional[int] = Field(
        20,
        ge=1,
        le=100,
        description="Number of recommendations to return"
    )
    
    @field_validator('new_ratings')
    @classmethod
    def validate_new_ratings(cls, v):
        """Validate new rating values."""
        if not v:
            raise ValueError("At least one new rating is required")
        
        for movie_id, rating in v.items():
            if not isinstance(movie_id, int) or movie_id <= 0:
                raise ValueError(f"Invalid movie ID: {movie_id}")
            if not isinstance(rating, (int, float)) or not (1.0 <= rating <= 5.0):
                raise ValueError(f"Rating must be between 1.0 and 5.0, got: {rating}")
        
        return v


class ErrorResponse(BaseModel):
    """Standardized error response model."""
    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: float = Field(..., description="Unix timestamp when error occurred")
    details: Optional[Dict] = Field(None, description="Additional error details")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "INSUFFICIENT_RATINGS",
                "message": "At least 15 ratings are required for recommendations",
                "status_code": 400,
                "timestamp": 1642234567.123,
                "details": {
                    "provided_ratings": 8,
                    "minimum_required": 15
                }
            }
        }
    )