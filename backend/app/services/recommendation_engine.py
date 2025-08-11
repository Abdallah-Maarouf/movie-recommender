"""
Core recommendation engine service.
Provides high-level interface for generating recommendations with caching and optimization.
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from app.models.recommendation import RecommendationResponse, RecommendationItem
from app.core.ml_models import ModelManager
from app.core.config import settings

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
        num_recommendations: int = 20
    ) -> RecommendationResponse:
        """
        Update recommendations with new ratings.
        
        Args:
            existing_ratings: Previously provided ratings
            new_ratings: New ratings to incorporate
            algorithm: Algorithm to use
            num_recommendations: Number of recommendations to return
            
        Returns:
            Updated RecommendationResponse
        """
        # Combine ratings
        all_ratings = {**existing_ratings, **new_ratings}
        
        logger.info(f"Updating recommendations: {len(existing_ratings)} existing + {len(new_ratings)} new = {len(all_ratings)} total")
        
        # Generate new recommendations
        response = await self.generate_recommendations(
            ratings=all_ratings,
            algorithm=algorithm,
            num_recommendations=num_recommendations,
            use_cache=False  # Don't use cache for updates
        )
        
        # Add update metadata
        response.metadata = response.metadata or {}
        response.metadata.update({
            'update_request': True,
            'existing_ratings_count': len(existing_ratings),
            'new_ratings_count': len(new_ratings),
            'total_ratings_count': len(all_ratings)
        })
        
        return response
    
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