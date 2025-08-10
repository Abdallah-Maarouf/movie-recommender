"""
Movie data models for request/response validation.
"""

from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional
from datetime import datetime


class MovieBase(BaseModel):
    """Base movie model with core attributes."""
    id: int = Field(..., description="Unique movie identifier")
    title: str = Field(..., min_length=1, max_length=500, description="Movie title")
    genres: List[str] = Field(..., description="List of movie genres")
    year: int = Field(..., ge=1900, le=2030, description="Release year")
    average_rating: float = Field(..., ge=0.0, le=5.0, description="Average user rating")
    rating_count: int = Field(..., ge=0, description="Number of ratings")
    
    @field_validator('genres')
    @classmethod
    def validate_genres(cls, v):
        """Ensure genres list is not empty and contains valid genres."""
        if not v:
            raise ValueError("Movie must have at least one genre")
        return v


class Movie(MovieBase):
    """Complete movie model with additional metadata."""
    poster_url: Optional[str] = Field(None, description="URL to movie poster image")
    description: Optional[str] = Field(None, max_length=2000, description="Movie description")
    director: Optional[str] = Field(None, max_length=200, description="Movie director")
    cast: Optional[List[str]] = Field(None, description="Main cast members")
    runtime: Optional[int] = Field(None, ge=1, description="Runtime in minutes")
    
    model_config = ConfigDict(
        json_encoders={
            datetime: lambda v: v.isoformat()
        },
        json_schema_extra={
            "example": {
                "id": 1,
                "title": "Toy Story",
                "genres": ["Animation", "Children's", "Comedy"],
                "year": 1995,
                "average_rating": 3.9,
                "rating_count": 2077,
                "poster_url": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
                "director": "John Lasseter",
                "cast": ["Tom Hanks", "Tim Allen", "Don Rickles"],
                "runtime": 81
            }
        }
    )


class MovieSummary(BaseModel):
    """Lightweight movie model for lists and summaries."""
    id: int
    title: str
    genres: List[str]
    year: int
    poster_url: Optional[str] = None
    average_rating: float
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": 1,
                "title": "Toy Story",
                "genres": ["Animation", "Children's", "Comedy"],
                "year": 1995,
                "poster_url": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                "average_rating": 3.9
            }
        }
    )


class InitialMoviesResponse(BaseModel):
    """Response model for initial movies endpoint."""
    movies: List[MovieSummary] = Field(..., description="List of movies for initial rating")
    total_count: int = Field(..., description="Total number of movies returned")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "movies": [
                    {
                        "id": 1,
                        "title": "Toy Story",
                        "genres": ["Animation", "Children's", "Comedy"],
                        "year": 1995,
                        "poster_url": "https://image.tmdb.org/t/p/w500/uXDfjJbdP4ijW5hWSBrPrlKpxab.jpg",
                        "average_rating": 3.9
                    }
                ],
                "total_count": 30
            }
        }
    )