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
from app.services.session_manager import get_session_manager
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
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Update recommendations with new ratings",
    description="Update existing recommendations by incorporating new user ratings with incremental computation and change tracking"
)
async def update_recommendations(
    request: UpdateRecommendationRequest,
    model_manager=Depends(get_model_manager)
):
    """
    Update recommendations with new ratings using incremental computation.
    
    Takes existing ratings plus new ratings and returns updated recommendations
    with detailed information about how the recommendations have changed,
    including position changes, new movies, and rating adjustments.
    
    Features:
    - Incremental computation for better performance
    - Session validation and rate limiting
    - Detailed change tracking and analytics
    - Recommendation explanation updates
    - Confidence scoring based on session analysis
    
    Args:
        request: Existing and new ratings with algorithm preferences
        
    Returns:
        Updated recommendations with comprehensive change indicators and analytics
    """
    start_time = time.time()
    
    try:
        logger.info(f"Updating recommendations with {len(request.new_ratings)} new ratings")
        
        # Get recommendation engine
        engine = await get_recommendation_engine(model_manager)
        
        # Extract previous recommendations if provided in metadata
        previous_recommendations = getattr(request, 'previous_recommendations', None)
        
        # Generate updated recommendations with session management
        recommendations = await engine.update_recommendations(
            existing_ratings=request.existing_ratings,
            new_ratings=request.new_ratings,
            algorithm=request.algorithm,
            num_recommendations=request.num_recommendations,
            previous_recommendations=previous_recommendations
        )
        
        # Add API-level metadata
        recommendations.metadata = recommendations.metadata or {}
        recommendations.metadata.update({
            'api_processing_time': time.time() - start_time,
            'endpoint': '/recommendations/update',
            'features_used': [
                'incremental_computation',
                'session_validation',
                'change_tracking',
                'analytics'
            ]
        })
        
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
                "status_code": 400,
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Error updating recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RECOMMENDATION_UPDATE_ERROR",
                "message": "Failed to update recommendations. Please try again later.",
                "status_code": 500,
                "timestamp": time.time()
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
        
        # Also clear session manager cache
        session_manager = get_session_manager()
        session_manager.cleanup_expired_data()
        
        logger.info("Recommendation and session cache cleared")
        
        return {
            "status": "success",
            "message": "All caches cleared successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "CACHE_CLEAR_ERROR",
                "message": "Failed to clear caches",
                "status_code": 500
            }
        )


@router.post(
    "/recommendations/validate-session",
    summary="Validate session data",
    description="Validate session data structure and check for suspicious patterns"
)
async def validate_session_data(
    session_data: dict
):
    """
    Validate session data for security and integrity.
    
    Checks for:
    - Valid data structure
    - Suspicious rating patterns
    - Rate limiting compliance
    - Data consistency
    
    Args:
        session_data: Session data to validate
        
    Returns:
        Validation result with details
    """
    try:
        session_manager = get_session_manager()
        
        is_valid, error_msg = session_manager.validate_session(session_data)
        
        if is_valid:
            # Generate session analytics
            ratings = session_data.get('ratings', {})
            session_analysis = session_manager.analyze_session(ratings, 'hybrid')
            
            return {
                "status": "valid",
                "message": "Session data is valid",
                "session_analysis": session_analysis,
                "timestamp": time.time()
            }
        else:
            return {
                "status": "invalid",
                "message": error_msg,
                "timestamp": time.time()
            }
            
    except Exception as e:
        logger.error(f"Error validating session: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SESSION_VALIDATION_ERROR",
                "message": "Failed to validate session data",
                "status_code": 500
            }
        )


@router.get(
    "/recommendations/session-stats",
    summary="Get session management statistics",
    description="Get comprehensive statistics about session management and analytics"
)
async def get_session_statistics():
    """
    Get session management statistics.
    
    Returns comprehensive statistics about:
    - Active sessions
    - Rate limiting status
    - Analytics summaries
    - Performance metrics
    """
    try:
        session_manager = get_session_manager()
        
        # Clean up expired data first
        session_manager.cleanup_expired_data()
        
        # Get comprehensive stats
        session_stats = session_manager.get_session_stats()
        
        return {
            "status": "success",
            "data": session_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting session stats: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "SESSION_STATS_ERROR",
                "message": "Failed to retrieve session statistics",
                "status_code": 500
            }
        )


@router.post(
    "/recommendations/analyze-ratings",
    summary="Analyze rating patterns",
    description="Analyze user rating patterns for insights and recommendations optimization"
)
async def analyze_rating_patterns(
    ratings_data: dict
):
    """
    Analyze user rating patterns for insights.
    
    Provides detailed analysis of:
    - Rating distribution and statistics
    - User behavior patterns
    - Recommendation confidence factors
    - Personalization insights
    
    Args:
        ratings_data: Dictionary containing user ratings
        
    Returns:
        Comprehensive rating pattern analysis
    """
    try:
        if 'ratings' not in ratings_data:
            raise ValueError("Missing 'ratings' field in request data")
        
        ratings = ratings_data['ratings']
        algorithm = ratings_data.get('algorithm', 'hybrid')
        
        session_manager = get_session_manager()
        
        # Validate ratings first
        session_data = {'ratings': ratings}
        is_valid, error_msg = session_manager.validate_session(session_data)
        
        if not is_valid:
            raise ValueError(f"Invalid ratings data: {error_msg}")
        
        # Perform comprehensive analysis
        pattern_analysis = session_manager.analytics.analyze_rating_pattern(ratings)
        confidence_score = session_manager.analytics.calculate_recommendation_confidence(ratings, algorithm)
        
        # Generate insights and recommendations
        insights = []
        
        if pattern_analysis.get('is_generous_rater'):
            insights.append("You tend to rate movies generously. Recommendations will focus on highly-rated content.")
        elif pattern_analysis.get('is_critical_rater'):
            insights.append("You have high standards for movies. Recommendations will emphasize critically acclaimed films.")
        
        if pattern_analysis.get('rating_variance') == 'high':
            insights.append("You have diverse taste in movies. Hybrid recommendations will work best for you.")
        else:
            insights.append("You have consistent rating patterns. Content-based recommendations may work well.")
        
        if len(ratings) >= 50:
            insights.append("With many ratings provided, collaborative filtering will be highly accurate.")
        elif len(ratings) < 20:
            insights.append("More ratings will improve recommendation accuracy. Consider rating more movies.")
        
        return {
            "status": "success",
            "data": {
                "pattern_analysis": pattern_analysis,
                "confidence_score": confidence_score,
                "insights": insights,
                "recommended_algorithm": "hybrid" if len(ratings) >= 25 else "content",
                "optimization_suggestions": [
                    "Rate movies from different genres for better diversity",
                    "Include both popular and niche movies in your ratings",
                    "Rate at least 30 movies for optimal collaborative filtering"
                ]
            },
            "timestamp": time.time()
        }
        
    except ValueError as e:
        logger.warning(f"Invalid rating analysis request: {e}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "INVALID_RATINGS_DATA",
                "message": str(e),
                "status_code": 400
            }
        )
    except Exception as e:
        logger.error(f"Error analyzing ratings: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "RATING_ANALYSIS_ERROR",
                "message": "Failed to analyze rating patterns",
                "status_code": 500
            }
        )