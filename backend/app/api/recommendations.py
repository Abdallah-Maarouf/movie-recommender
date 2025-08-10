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
    model_manager=Depends(get_model_manager),
    data_loader=Depends(get_data_loader)
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
        
        # Validate that we have the required models
        if not model_manager.is_ready():
            logger.warning("ML models not ready, using fallback recommendations")
            # For now, we'll implement a simple fallback
            # This will be properly implemented in task 3.3
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "MODELS_NOT_READY",
                    "message": "Recommendation models are not ready. Please try again later.",
                    "status_code": 503
                }
            )
        
        # Generate recommendations using the specified algorithm
        recommendations = await model_manager.generate_recommendations(
            ratings=request.ratings,
            algorithm=request.algorithm,
            num_recommendations=request.num_recommendations
        )
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated {len(recommendations.recommendations)} recommendations in {processing_time:.3f}s")
        
        # Add processing time to response
        recommendations.processing_time = processing_time
        
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
    model_manager=Depends(get_model_manager),
    data_loader=Depends(get_data_loader)
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
        
        # Combine existing and new ratings
        all_ratings = {**request.existing_ratings, **request.new_ratings}
        
        logger.info(f"Total ratings after update: {len(all_ratings)}")
        
        # Validate minimum rating requirements
        if len(all_ratings) < 15:
            raise ValueError(f"At least 15 total ratings required, got {len(all_ratings)}")
        
        # Generate updated recommendations
        recommendations = await model_manager.generate_recommendations(
            ratings=all_ratings,
            algorithm=request.algorithm,
            num_recommendations=request.num_recommendations
        )
        
        processing_time = time.time() - start_time
        recommendations.processing_time = processing_time
        
        logger.info(f"Updated recommendations generated in {processing_time:.3f}s")
        
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