"""
Movie-related API endpoints with TMDB integration.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Optional
import logging
import time

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
    description="Returns a curated list of 30 diverse movies with posters for users to rate initially"
)
async def get_initial_movies(data_loader=Depends(get_data_loader)):
    """
    Get initial movies for the rating interface.
    
    Returns 30 carefully selected movies covering different genres and decades
    to help establish user preferences for the recommendation system.
    All movies include poster images from TMDB.
    """
    start_time = time.time()
    
    try:
        logger.info("Fetching initial movies for rating interface")
        
        # Get initial movies from data loader (includes TMDB posters)
        movies = await data_loader.get_initial_movies()
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully retrieved {len(movies)} initial movies in {processing_time:.3f}s")
        
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
                "status_code": 500,
                "timestamp": time.time()
            }
        )


@router.get(
    "/movies/search",
    response_model=List[Movie],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid search parameters"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Search movies",
    description="Search for movies by title, genre, or year"
)
async def search_movies(
    query: Optional[str] = Query(None, description="Search query for movie title"),
    genre: Optional[str] = Query(None, description="Filter by genre"),
    year: Optional[int] = Query(None, description="Filter by release year"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
    data_loader=Depends(get_data_loader)
):
    """
    Search for movies based on various criteria.
    
    Args:
        query: Search term for movie titles
        genre: Filter by specific genre
        year: Filter by release year
        limit: Maximum number of results to return
        
    Returns:
        List of movies matching the search criteria
    """
    start_time = time.time()
    
    try:
        logger.info(f"Searching movies with query='{query}', genre='{genre}', year={year}")
        
        if not any([query, genre, year]):
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_SEARCH_PARAMETERS",
                    "message": "At least one search parameter (query, genre, or year) is required",
                    "status_code": 400,
                    "timestamp": time.time()
                }
            )
        
        # Get all movies
        all_movies = await data_loader.get_all_movies()
        
        # Filter movies based on search criteria
        filtered_movies = []
        
        for movie in all_movies.values():
            # Check query match (case-insensitive title search)
            if query and query.lower() not in movie.title.lower():
                continue
            
            # Check genre match
            if genre and genre.lower() not in [g.lower() for g in movie.genres]:
                continue
            
            # Check year match
            if year and movie.year != year:
                continue
            
            filtered_movies.append(movie)
            
            # Limit results
            if len(filtered_movies) >= limit:
                break
        
        processing_time = time.time() - start_time
        logger.info(f"Found {len(filtered_movies)} movies matching search criteria in {processing_time:.3f}s")
        
        return filtered_movies
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching movies: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SEARCH_ERROR",
                "message": "Failed to search movies. Please try again later.",
                "status_code": 500,
                "timestamp": time.time()
            }
        )


@router.get(
    "/movies/stats",
    summary="Get data statistics",
    description="Returns statistics about the loaded movie data"
)
async def get_data_stats(data_loader=Depends(get_data_loader)):
    """
    Get statistics about the loaded movie data.
    
    Returns:
        Statistics including total movies, movies with posters, etc.
    """
    try:
        logger.info("Fetching data statistics")
        
        stats = data_loader.get_data_summary()
        
        logger.info("Successfully retrieved data statistics")
        return stats
        
    except Exception as e:
        logger.error(f"Error fetching data statistics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATS_ERROR",
                "message": "Failed to load data statistics. Please try again later.",
                "status_code": 500,
                "timestamp": time.time()
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
    description="Returns detailed information about a specific movie including TMDB poster"
)
async def get_movie(movie_id: int, data_loader=Depends(get_data_loader)):
    """
    Get detailed information about a specific movie.
    
    Args:
        movie_id: The unique identifier of the movie
        
    Returns:
        Complete movie information including metadata and poster URL from TMDB
    """
    start_time = time.time()
    
    try:
        logger.info(f"Fetching details for movie ID: {movie_id}")
        
        # Get movie details from data loader (includes TMDB poster fetching)
        movie = await data_loader.get_movie_by_id(movie_id)
        
        if not movie:
            logger.warning(f"Movie not found: {movie_id}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "MOVIE_NOT_FOUND",
                    "message": f"Movie with ID {movie_id} not found",
                    "status_code": 404,
                    "timestamp": time.time()
                }
            )
        
        processing_time = time.time() - start_time
        logger.info(f"Successfully retrieved movie: {movie.title} in {processing_time:.3f}s")
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
                "status_code": 500,
                "timestamp": time.time()
            }
        )