"""
Core recommendation engine service.
Provides high-level interface for generating recommendations with caching and optimization.
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np

from app.models.recommendation import RecommendationResponse, RecommendationItem
from app.core.ml_models import ModelManager
from app.core.config import settings
from app.services.session_manager import get_session_manager

logger = logging.getLogger(__name__)


class RecommendationCache:
    """Simple in-memory cache for recommendation responses."""
    
    def __init__(self, ttl: int = 3600):
        self.cache = {}
        self.ttl = ttl
    
    def _generate_key(self, ratings: Dict[int, float], algorithm: str, num_recommendations: int) -> str:
        """Generate cache key from request parameters."""
        # Sort ratings for consistent key generation
        sorted_ratings = dict(sorted(ratings.items()))
        key_data = {
            'ratings': sorted_ratings,
            'algorithm': algorithm,
            'num_recommendations': num_recommendations
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, ratings: Dict[int, float], algorithm: str, num_recommendations: int) -> Optional[RecommendationResponse]:
        """Get cached recommendation response."""
        if not settings.ENABLE_CACHING:
            return None
        
        key = self._generate_key(ratings, algorithm, num_recommendations)
        
        if key in self.cache:
            cached_item = self.cache[key]
            # Check if cache entry is still valid
            if time.time() - cached_item['timestamp'] < self.ttl:
                logger.info(f"Cache hit for key: {key[:8]}...")
                return cached_item['response']
            else:
                # Remove expired entry
                del self.cache[key]
        
        return None
    
    def set(self, ratings: Dict[int, float], algorithm: str, num_recommendations: int, response: RecommendationResponse):
        """Cache recommendation response."""
        if not settings.ENABLE_CACHING:
            return
        
        key = self._generate_key(ratings, algorithm, num_recommendations)
        self.cache[key] = {
            'response': response,
            'timestamp': time.time()
        }
        logger.info(f"Cached response for key: {key[:8]}...")
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Recommendation cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = sum(
            1 for item in self.cache.values() 
            if current_time - item['timestamp'] < self.ttl
        )
        
        return {
            'total_entries': len(self.cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self.cache) - valid_entries,
            'ttl': self.ttl
        }


class RecommendationEngine:
    """
    High-level recommendation engine that orchestrates model inference,
    caching, and performance optimization.
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.cache = RecommendationCache(ttl=settings.CACHE_TTL)
        self.request_count = 0
        self.total_processing_time = 0.0
    
    async def generate_recommendations(
        self,
        ratings: Dict[int, float],
        algorithm: str = "hybrid",
        num_recommendations: int = 20,
        use_cache: bool = True
    ) -> RecommendationResponse:
        """
        Generate recommendations with caching and performance optimization.
        
        Args:
            ratings: User ratings dictionary
            algorithm: Algorithm to use (collaborative, content, hybrid)
            num_recommendations: Number of recommendations to return
            use_cache: Whether to use cached results
            
        Returns:
            RecommendationResponse with recommendations and metadata
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            # Check cache first
            if use_cache:
                cached_response = self.cache.get(ratings, algorithm, num_recommendations)
                if cached_response:
                    # Update processing time to include cache lookup
                    cached_response.processing_time = time.time() - start_time
                    cached_response.metadata = cached_response.metadata or {}
                    cached_response.metadata['from_cache'] = True
                    return cached_response
            
            # Validate input
            self._validate_input(ratings, algorithm, num_recommendations)
            
            # Generate recommendations using model manager
            response = await self.model_manager.generate_recommendations(
                ratings=ratings,
                algorithm=algorithm,
                num_recommendations=num_recommendations
            )
            
            # Add diversity if requested
            response = await self._add_diversity(response, ratings)
            
            # Cache the response
            if use_cache:
                self.cache.set(ratings, algorithm, num_recommendations, response)
            
            # Update statistics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            
            # Add engine metadata
            response.metadata = response.metadata or {}
            response.metadata.update({
                'from_cache': False,
                'request_id': self.request_count,
                'engine_processing_time': processing_time
            })
            
            logger.info(f"Generated {len(response.recommendations)} recommendations in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in recommendation engine: {e}")
            raise
    
    def _validate_input(self, ratings: Dict[int, float], algorithm: str, num_recommendations: int):
        """Validate input parameters."""
        if not ratings:
            raise ValueError("Ratings dictionary cannot be empty")
        
        if len(ratings) < settings.MIN_RATINGS_REQUIRED:
            raise ValueError(f"At least {settings.MIN_RATINGS_REQUIRED} ratings required")
        
        if algorithm not in ["collaborative", "content", "hybrid"]:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        
        if not (1 <= num_recommendations <= 100):
            raise ValueError("Number of recommendations must be between 1 and 100")
        
        # Validate rating values
        for movie_id, rating in ratings.items():
            if not isinstance(movie_id, int) or movie_id <= 0:
                raise ValueError(f"Invalid movie ID: {movie_id}")
            if not isinstance(rating, (int, float)) or not (1.0 <= rating <= 5.0):
                raise ValueError(f"Rating must be between 1.0 and 5.0, got: {rating}")
    
    async def _add_diversity(self, response: RecommendationResponse, ratings: Dict[int, float]) -> RecommendationResponse:
        """Add diversity to recommendations to avoid over-specialization."""
        if len(response.recommendations) <= 5:
            return response  # No need to diversify small lists
        
        try:
            # Group recommendations by genre
            genre_groups = {}
            for rec in response.recommendations:
                for genre in rec.movie.genres:
                    if genre not in genre_groups:
                        genre_groups[genre] = []
                    genre_groups[genre].append(rec)
            
            # Ensure diversity by limiting recommendations per genre
            max_per_genre = max(2, len(response.recommendations) // len(genre_groups))
            diversified_recs = []
            used_movies = set()
            
            # First pass: take top recommendations from each genre
            for genre, recs in genre_groups.items():
                recs.sort(key=lambda x: x.predicted_rating * x.confidence, reverse=True)
                count = 0
                for rec in recs:
                    if rec.movie.id not in used_movies and count < max_per_genre:
                        diversified_recs.append(rec)
                        used_movies.add(rec.movie.id)
                        count += 1
            
            # Second pass: fill remaining slots with highest-rated recommendations
            remaining_slots = len(response.recommendations) - len(diversified_recs)
            if remaining_slots > 0:
                remaining_recs = [
                    rec for rec in response.recommendations 
                    if rec.movie.id not in used_movies
                ]
                remaining_recs.sort(key=lambda x: x.predicted_rating * x.confidence, reverse=True)
                diversified_recs.extend(remaining_recs[:remaining_slots])
            
            # Update response
            response.recommendations = diversified_recs[:len(response.recommendations)]
            response.metadata = response.metadata or {}
            response.metadata['diversity_applied'] = True
            
        except Exception as e:
            logger.warning(f"Error applying diversity: {e}")
            # Return original response if diversity fails
        
        return response
    
    async def update_recommendations(
        self,
        existing_ratings: Dict[int, float],
        new_ratings: Dict[int, float],
        algorithm: str = "hybrid",
        num_recommendations: int = 20,
        previous_recommendations: Optional[List[Dict]] = None
    ) -> RecommendationResponse:
        """
        Update recommendations with new ratings using incremental computation.
        
        Args:
            existing_ratings: Previously provided ratings
            new_ratings: New ratings to incorporate
            algorithm: Algorithm to use
            num_recommendations: Number of recommendations to return
            previous_recommendations: Previous recommendations for delta calculation
            
        Returns:
            Updated RecommendationResponse with change indicators
        """
        start_time = time.time()
        session_manager = get_session_manager()
        
        # Validate and merge ratings
        all_ratings = {**existing_ratings, **new_ratings}
        
        # Validate session data
        session_data = {'ratings': all_ratings}
        is_valid, error_msg = session_manager.validate_session(session_data)
        if not is_valid:
            raise ValueError(f"Invalid session data: {error_msg}")
        
        logger.info(f"Updating recommendations: {len(existing_ratings)} existing + {len(new_ratings)} new = {len(all_ratings)} total")
        
        # Check if we can use incremental computation
        use_incremental = (
            len(new_ratings) <= 5 and  # Small number of new ratings
            len(existing_ratings) >= 15 and  # Sufficient existing data
            algorithm in ["collaborative", "hybrid"]  # Supported algorithms
        )
        
        if use_incremental:
            try:
                response = await self._incremental_update(
                    existing_ratings, new_ratings, algorithm, num_recommendations
                )
            except Exception as e:
                logger.warning(f"Incremental update failed, falling back to full computation: {e}")
                use_incremental = False
        
        if not use_incremental:
            # Generate new recommendations using full computation
            response = await self.generate_recommendations(
                ratings=all_ratings,
                algorithm=algorithm,
                num_recommendations=num_recommendations,
                use_cache=False  # Don't use cache for updates
            )
        
        # Calculate recommendation changes
        recommendation_delta = None
        if previous_recommendations:
            recommendation_delta = session_manager.calculate_recommendation_delta(
                previous_recommendations, 
                [rec.dict() for rec in response.recommendations]
            )
        
        # Analyze session for insights
        session_analysis = session_manager.analyze_session(all_ratings, algorithm)
        
        # Add comprehensive update metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            'update_request': True,
            'existing_ratings_count': len(existing_ratings),
            'new_ratings_count': len(new_ratings),
            'total_ratings_count': len(all_ratings),
            'incremental_computation_used': use_incremental,
            'session_analysis': session_analysis,
            'recommendation_delta': recommendation_delta,
            'update_processing_time': time.time() - start_time
        })
        
        return response
    
    async def _incremental_update(
        self,
        existing_ratings: Dict[int, float],
        new_ratings: Dict[int, float],
        algorithm: str,
        num_recommendations: int
    ) -> RecommendationResponse:
        """
        Perform incremental recommendation update for better performance.
        
        This method attempts to update recommendations without full recomputation
        by analyzing the impact of new ratings on the existing user profile.
        """
        start_time = time.time()
        
        # Get existing recommendations for comparison
        existing_response = await self.generate_recommendations(
            ratings=existing_ratings,
            algorithm=algorithm,
            num_recommendations=num_recommendations * 2,  # Get more for better selection
            use_cache=True
        )
        
        # Analyze new ratings impact
        new_rating_values = list(new_ratings.values())
        existing_rating_values = list(existing_ratings.values())
        
        # Calculate rating shift
        new_mean = np.mean(new_rating_values)
        existing_mean = np.mean(existing_rating_values)
        rating_shift = new_mean - existing_mean
        
        # Adjust predictions based on rating shift and new preferences
        adjusted_recommendations = []
        
        for rec in existing_response.recommendations:
            # Adjust prediction based on rating shift
            adjusted_rating = rec.predicted_rating + (rating_shift * 0.1)
            
            # Boost recommendations similar to highly rated new movies
            similarity_boost = 0.0
            for new_movie_id, new_rating in new_ratings.items():
                if new_rating >= 4.0:  # Highly rated new movie
                    # Calculate genre similarity (simplified)
                    if hasattr(rec.movie, 'genres') and new_movie_id != rec.movie.id:
                        # This would need actual movie data for proper similarity
                        similarity_boost += 0.05  # Placeholder boost
            
            adjusted_rating = min(5.0, max(1.0, adjusted_rating + similarity_boost))
            
            # Update confidence based on new data
            confidence_adjustment = min(0.1, len(new_ratings) * 0.02)
            adjusted_confidence = min(0.95, rec.confidence + confidence_adjustment)
            
            adjusted_recommendations.append(RecommendationItem(
                movie=rec.movie,
                predicted_rating=round(adjusted_rating, 2),
                confidence=round(adjusted_confidence, 2),
                explanation=f"{rec.explanation} (updated with new preferences)",
                algorithm_used=f"{algorithm}_incremental"
            ))
        
        # Sort by adjusted predictions and select top N
        adjusted_recommendations.sort(
            key=lambda x: x.predicted_rating * x.confidence, 
            reverse=True
        )
        final_recommendations = adjusted_recommendations[:num_recommendations]
        
        # Calculate overall confidence
        overall_confidence = np.mean([rec.confidence for rec in final_recommendations])
        
        processing_time = time.time() - start_time
        
        return RecommendationResponse(
            recommendations=final_recommendations,
            total_ratings=len(existing_ratings) + len(new_ratings),
            algorithm_used=f"{algorithm}_incremental",
            confidence_score=round(overall_confidence, 2),
            processing_time=processing_time,
            metadata={
                'incremental_update': True,
                'rating_shift': round(rating_shift, 2),
                'base_recommendations': len(existing_response.recommendations)
            }
        )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get recommendation engine statistics."""
        avg_processing_time = (
            self.total_processing_time / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'total_requests': self.request_count,
            'total_processing_time': self.total_processing_time,
            'average_processing_time': avg_processing_time,
            'cache_stats': self.cache.get_stats(),
            'model_ready': self.model_manager.is_ready()
        }
    
    def clear_cache(self):
        """Clear recommendation cache."""
        self.cache.clear()
    
    async def warm_up(self):
        """Warm up the recommendation engine by loading models."""
        logger.info("Warming up recommendation engine...")
        await self.model_manager.load_models()
        logger.info("Recommendation engine ready")


# Global recommendation engine instance
_recommendation_engine: Optional[RecommendationEngine] = None


async def get_recommendation_engine(model_manager: ModelManager) -> RecommendationEngine:
    """Get or create recommendation engine instance."""
    global _recommendation_engine
    
    if _recommendation_engine is None:
        _recommendation_engine = RecommendationEngine(model_manager)
        await _recommendation_engine.warm_up()
    
    return _recommendation_engine