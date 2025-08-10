"""
Movie-related API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List
import logging

from app.models.movie import InitialMoviesResponse, Movie
from app.models.recommendation import ErrorResponse
from app.utils.data_loader import get_data_loader
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/movies/initial",
    response_model=InitialMoviesResponse,
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get initial movies for rating",
    description="Returns a curated list of 30 diverse movies for users to rate initially"
)
async def get_initial_movies(data_loader=Depends(get_data_loader)):
    """
    Get initial movies for the rating interface.
    
    Returns 30 carefully selected movies covering different genres and decades
    to help establish user preferences for the recommendation system.
    """
    try:
        logger.info("Fetching initial movies for rating interface")
        
        # Get initial movies from data loader
        movies = await data_loader.get_initial_movies()
        
        logger.info(f"Successfully retrieved {len(movies)} initial movies")
        
        return InitialMoviesResponse(
            movies=movies,
            total_count=len(movies)
        )
        
    except Exception as e:
        logger.error(f"Error fetching initial movies: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATA_LOADING_ERROR",
                "message": "Failed to load initial movies. Please try again later.",
                "status_code": 500
            }
        )


@router.get(
    "/movies/{movie_id}",
    response_model=Movie,
    responses={
        404: {"model": ErrorResponse, "description": "Movie not found"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Get movie details",
    description="Returns detailed information about a specific movie"
)
async def get_movie(movie_id: int, data_loader=Depends(get_data_loader)):
    """
    Get detailed information about a specific movie.
    
    Args:
        movie_id: The unique identifier of the movie
        
    Returns:
        Complete movie information including metadata and poster URL
    """
    try:
        logger.info(f"Fetching details for movie ID: {movie_id}")
        
        # Get movie details from data loader
        movie = await data_loader.get_movie_by_id(movie_id)
        
        if not movie:
            logger.warning(f"Movie not found: {movie_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "MOVIE_NOT_FOUND",
                    "message": f"Movie with ID {movie_id} not found",
                    "status_code": 404
                }
            )
        
        logger.info(f"Successfully retrieved movie: {movie.title}")
        return movie
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching movie {movie_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "DATA_LOADING_ERROR",
                "message": "Failed to load movie details. Please try again later.",
                "status_code": 500
            }
        )