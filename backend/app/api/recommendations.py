"""
Recommendation-related API endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List
import logging
import time

from app.models.recommendation import (
    RatingRequest, 
    RecommendationResponse, 
    UpdateRecommendationRequest,
    ErrorResponse
)
from app.core.ml_models import get_model_manager
from app.services.recommendation_engine import get_recommendation_engine
from app.utils.data_loader import get_data_loader

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/recommendations",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Generate movie recommendations",
    description="Generate personalized movie recommendations based on user ratings"
)
async def generate_recommendations(
    request: RatingRequest,
    model_manager=Depends(get_model_manager)
):
    """
    Generate personalized movie recommendations.
    
    Takes user ratings and returns personalized recommendations using
    collaborative filtering, content-based filtering, or hybrid approaches.
    
    Args:
        request: User ratings and algorithm preferences
        
    Returns:
        List of recommended movies with predictions and explanations
    """
    start_time = time.time()
    
    try:
        logger.info(f"Generating recommendations for {len(request.ratings)} ratings using {request.algorithm}")
        
        # Get recommendation engine
        engine = await get_recommendation_engine(model_manager)
        
        # Generate recommendations using the engine
        recommendations = await engine.generate_recommendations(
            ratings=request.ratings,
            algorithm=request.algorithm,
            num_recommendations=request.num_recommendations
        )
        
        logger.info(f"Generated {len(recommendations.recommendations)} recommendations in {recommendations.processing_time:.3f}s")
        
        return recommendations
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid request data: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_REQUEST",
                "message": str(e),
                "status_code": 400
            }
        )
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RECOMMENDATION_ERROR",
                "message": "Failed to generate recommendations. Please try again later.",
                "status_code": 500
            }
        )


@router.post(
    "/recommendations/update",
    response_model=RecommendationResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Update recommendations with new ratings",
    description="Update existing recommendations by incorporating new user ratings"
)
async def update_recommendations(
    request: UpdateRecommendationRequest,
    model_manager=Depends(get_model_manager)
):
    """
    Update recommendations with new ratings.
    
    Takes existing ratings plus new ratings and returns updated recommendations
    with information about how the recommendations have changed.
    
    Args:
        request: Existing and new ratings with algorithm preferences
        
    Returns:
        Updated recommendations with change indicators
    """
    start_time = time.time()
    
    try:
        logger.info(f"Updating recommendations with {len(request.new_ratings)} new ratings")
        
        # Get recommendation engine
        engine = await get_recommendation_engine(model_manager)
        
        # Generate updated recommendations
        recommendations = await engine.update_recommendations(
            existing_ratings=request.existing_ratings,
            new_ratings=request.new_ratings,
            algorithm=request.algorithm,
            num_recommendations=request.num_recommendations
        )
        
        logger.info(f"Updated recommendations generated in {recommendations.processing_time:.3f}s")
        
        return recommendations
        
    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Invalid update request: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_UPDATE_REQUEST",
                "message": str(e),
                "status_code": 400
            }
        )
    except Exception as e:
        logger.error(f"Error updating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RECOMMENDATION_UPDATE_ERROR",
                "message": "Failed to update recommendations. Please try again later.",
                "status_code": 500
            }
        )

@router.get(
    "/recommendations/stats",
    summary="Get recommendation engine statistics",
    description="Get performance statistics and status information about the recommendation engine"
)
async def get_recommendation_stats(
    model_manager=Depends(get_model_manager)
):
    """
    Get recommendation engine statistics.
    
    Returns performance metrics, cache statistics, and model status information.
    """
    try:
        engine = await get_recommendation_engine(model_manager)
        stats = engine.get_engine_stats()
        
        # Add model validation status
        model_status = model_manager._validate_models()
        stats['model_status'] = model_status
        
        return {
            "status": "success",
            "data": stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting recommendation stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "STATS_ERROR",
                "message": "Failed to retrieve recommendation statistics",
                "status_code": 500
            }
        )


@router.post(
    "/recommendations/cache/clear",
    summary="Clear recommendation cache",
    description="Clear the recommendation cache to force fresh recommendations"
)
async def clear_recommendation_cache(
    model_manager=Depends(get_model_manager)
):
    """
    Clear the recommendation cache.
    
    Forces all subsequent recommendations to be generated fresh
    instead of using cached results.
    """
    try:
        engine = await get_recommendation_engine(model_manager)
        engine.clear_cache()
        
        logger.info("Recommendation cache cleared")
        
        return {
            "status": "success",
            "message": "Recommendation cache cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CACHE_CLEAR_ERROR",
                "message": "Failed to clear recommendation cache",
                "status_code": 500
            }
        )