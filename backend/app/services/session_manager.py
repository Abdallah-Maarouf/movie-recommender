"""
Session management service for stateless session handling and analytics.
Provides session validation, analytics, and temporary storage optimization.
"""

import logging
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class SessionAnalytics:
    """Analytics for user rating patterns and recommendation accuracy."""
    
    def __init__(self):
        self.rating_patterns = {}
        self.recommendation_feedback = {}
        self.session_metrics = {}
    
    def analyze_rating_pattern(self, ratings: Dict[int, float]) -> Dict[str, Any]:
        """Analyze user rating patterns for insights."""
        if not ratings:
            return {}
        
        rating_values = list(ratings.values())
        
        # Basic statistics
        mean_rating = np.mean(rating_values)
        std_rating = np.std(rating_values)
        rating_range = max(rating_values) - min(rating_values)
        
        # Rating distribution
        rating_counts = defaultdict(int)
        for rating in rating_values:
            rating_counts[int(rating)] += 1
        
        # Genre preferences (would need movie data)
        # This is a placeholder for genre analysis
        
        return {
            'total_ratings': len(ratings),
            'mean_rating': round(float(mean_rating), 2),
            'std_rating': round(float(std_rating), 2),
            'rating_range': float(rating_range),
            'rating_distribution': dict(rating_counts),
            'is_generous_rater': bool(mean_rating > 3.5),
            'is_critical_rater': bool(mean_rating < 2.5),
            'rating_variance': 'high' if std_rating > 1.0 else 'low'
        }
    
    def calculate_recommendation_confidence(
        self, 
        ratings: Dict[int, float], 
        algorithm: str
    ) -> float:
        """Calculate confidence score for recommendations based on user profile."""
        if len(ratings) < settings.MIN_RATINGS_REQUIRED:
            return 0.3  # Low confidence for insufficient data
        
        # Base confidence increases with more ratings
        base_confidence = min(0.9, 0.4 + (len(ratings) - 15) * 0.02)
        
        # Adjust based on rating patterns
        pattern = self.analyze_rating_pattern(ratings)
        
        # Higher confidence for users with varied ratings
        if pattern.get('rating_variance') == 'high':
            base_confidence += 0.1
        
        # Lower confidence for extreme raters
        if pattern.get('is_generous_rater') or pattern.get('is_critical_rater'):
            base_confidence -= 0.05
        
        # Algorithm-specific adjustments
        if algorithm == 'collaborative' and len(ratings) > 30:
            base_confidence += 0.05
        elif algorithm == 'content' and pattern.get('rating_variance') == 'high':
            base_confidence += 0.05
        
        return max(0.1, min(0.95, base_confidence))
    
    def track_session_metrics(self, session_id: str, metrics: Dict[str, Any]):
        """Track session-level metrics for analysis."""
        self.session_metrics[session_id] = {
            **metrics,
            'timestamp': time.time()
        }
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of all analytics data."""
        return {
            'total_sessions': len(self.session_metrics),
            'rating_patterns_analyzed': len(self.rating_patterns),
            'recommendation_feedback_count': len(self.recommendation_feedback),
            'last_updated': time.time()
        }


class SessionValidator:
    """Validates session data and prevents abuse."""
    
    def __init__(self):
        self.rate_limits = defaultdict(deque)
        self.suspicious_sessions = set()
    
    def validate_session_data(self, session_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate session data structure and content."""
        try:
            # Check required fields
            if 'ratings' not in session_data:
                return False, "Missing ratings data"
            
            ratings = session_data['ratings']
            if not isinstance(ratings, dict):
                return False, "Ratings must be a dictionary"
            
            # Validate ratings
            for movie_id, rating in ratings.items():
                try:
                    movie_id = int(movie_id)
                    rating = float(rating)
                except (ValueError, TypeError):
                    return False, f"Invalid rating data: {movie_id}={rating}"
                
                if movie_id <= 0:
                    return False, f"Invalid movie ID: {movie_id}"
                
                if not (1.0 <= rating <= 5.0):
                    return False, f"Rating must be between 1.0 and 5.0: {rating}"
            
            # Check for suspicious patterns
            if self._detect_suspicious_patterns(ratings):
                return False, "Suspicious rating patterns detected"
            
            return True, None
            
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False, f"Session validation failed: {str(e)}"
    
    def _detect_suspicious_patterns(self, ratings: Dict[int, float]) -> bool:
        """Detect suspicious rating patterns that might indicate abuse."""
        if len(ratings) > 1000:  # Too many ratings
            return True
        
        rating_values = list(ratings.values())
        
        # All identical ratings (except for very small sets)
        if len(set(rating_values)) == 1 and len(rating_values) > 10:
            return True
        
        # Sequential movie IDs with identical ratings (bot-like behavior)
        movie_ids = sorted(ratings.keys())
        if len(movie_ids) > 20:
            sequential_count = 0
            for i in range(1, len(movie_ids)):
                if movie_ids[i] == movie_ids[i-1] + 1:
                    sequential_count += 1
            
            if sequential_count > len(movie_ids) * 0.8:  # 80% sequential
                return True
        
        return False
    
    def check_rate_limit(self, session_id: str, max_requests: int = 100, window_minutes: int = 60) -> bool:
        """Check if session is within rate limits."""
        current_time = time.time()
        window_start = current_time - (window_minutes * 60)
        
        # Clean old requests
        session_requests = self.rate_limits[session_id]
        while session_requests and session_requests[0] < window_start:
            session_requests.popleft()
        
        # Check limit
        if len(session_requests) >= max_requests:
            logger.warning(f"Rate limit exceeded for session: {session_id}")
            return False
        
        # Add current request
        session_requests.append(current_time)
        return True
    
    def mark_suspicious(self, session_id: str):
        """Mark a session as suspicious."""
        self.suspicious_sessions.add(session_id)
        logger.warning(f"Session marked as suspicious: {session_id}")
    
    def is_suspicious(self, session_id: str) -> bool:
        """Check if session is marked as suspicious."""
        return session_id in self.suspicious_sessions


class SessionManager:
    """
    Stateless session manager with analytics and performance optimization.
    Handles session validation, analytics, and temporary storage.
    """
    
    def __init__(self):
        self.analytics = SessionAnalytics()
        self.validator = SessionValidator()
        self.temp_storage = {}  # Temporary storage for performance optimization
        self.session_cache = {}  # Cache for session computations
    
    def generate_session_id(self, ratings: Dict[int, float]) -> str:
        """Generate deterministic session ID from ratings."""
        # Sort ratings for consistent ID generation
        sorted_ratings = dict(sorted(ratings.items()))
        session_data = json.dumps(sorted_ratings, sort_keys=True)
        return hashlib.sha256(session_data.encode()).hexdigest()[:16]
    
    def validate_session(self, session_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate session data and check rate limits."""
        # Validate data structure
        is_valid, error_msg = self.validator.validate_session_data(session_data)
        if not is_valid:
            return False, error_msg
        
        # Generate session ID for rate limiting
        session_id = self.generate_session_id(session_data['ratings'])
        
        # Check if session is suspicious
        if self.validator.is_suspicious(session_id):
            return False, "Session blocked due to suspicious activity"
        
        # Check rate limits
        if not self.validator.check_rate_limit(session_id):
            self.validator.mark_suspicious(session_id)
            return False, "Rate limit exceeded"
        
        return True, None
    
    def analyze_session(self, ratings: Dict[int, float], algorithm: str) -> Dict[str, Any]:
        """Analyze session for insights and confidence scoring."""
        session_id = self.generate_session_id(ratings)
        
        # Analyze rating patterns
        pattern_analysis = self.analytics.analyze_rating_pattern(ratings)
        
        # Calculate confidence
        confidence = self.analytics.calculate_recommendation_confidence(ratings, algorithm)
        
        # Session metrics
        session_metrics = {
            'session_id': session_id,
            'total_ratings': len(ratings),
            'algorithm_used': algorithm,
            'confidence_score': confidence,
            'pattern_analysis': pattern_analysis,
            'timestamp': time.time()
        }
        
        # Track metrics
        self.analytics.track_session_metrics(session_id, session_metrics)
        
        return session_metrics
    
    def optimize_session_data(self, ratings: Dict[int, float]) -> Dict[str, Any]:
        """Optimize session data for performance."""
        session_id = self.generate_session_id(ratings)
        
        # Check if we have cached optimizations
        if session_id in self.session_cache:
            cached_data = self.session_cache[session_id]
            if time.time() - cached_data['timestamp'] < 300:  # 5 minute cache
                return cached_data['optimizations']
        
        # Perform optimizations
        optimizations = {
            'sorted_ratings': dict(sorted(ratings.items())),
            'rating_vector': list(ratings.values()),
            'movie_ids': list(ratings.keys()),
            'mean_rating': np.mean(list(ratings.values())),
            'rating_count': len(ratings)
        }
        
        # Cache optimizations
        self.session_cache[session_id] = {
            'optimizations': optimizations,
            'timestamp': time.time()
        }
        
        return optimizations
    
    def calculate_recommendation_delta(
        self, 
        old_recommendations: List[Dict], 
        new_recommendations: List[Dict]
    ) -> Dict[str, Any]:
        """Calculate changes between old and new recommendations."""
        if not old_recommendations:
            return {
                'new_movies': len(new_recommendations),
                'removed_movies': 0,
                'position_changes': [],
                'rating_changes': [],
                'change_summary': 'Initial recommendations generated'
            }
        
        # Create lookup dictionaries
        old_lookup = {rec['movie']['id']: rec for rec in old_recommendations}
        new_lookup = {rec['movie']['id']: rec for rec in new_recommendations}
        
        # Calculate changes
        old_ids = set(old_lookup.keys())
        new_ids = set(new_lookup.keys())
        
        new_movies = new_ids - old_ids
        removed_movies = old_ids - new_ids
        common_movies = old_ids & new_ids
        
        position_changes = []
        rating_changes = []
        
        # Analyze position and rating changes for common movies
        for movie_id in common_movies:
            old_rec = old_lookup[movie_id]
            new_rec = new_lookup[movie_id]
            
            # Position change
            old_pos = next(i for i, rec in enumerate(old_recommendations) if rec['movie']['id'] == movie_id)
            new_pos = next(i for i, rec in enumerate(new_recommendations) if rec['movie']['id'] == movie_id)
            
            if old_pos != new_pos:
                position_changes.append({
                    'movie_id': movie_id,
                    'movie_title': new_rec['movie']['title'],
                    'old_position': old_pos + 1,
                    'new_position': new_pos + 1,
                    'change': new_pos - old_pos
                })
            
            # Rating change
            old_rating = old_rec['predicted_rating']
            new_rating = new_rec['predicted_rating']
            
            if abs(old_rating - new_rating) > 0.1:
                rating_changes.append({
                    'movie_id': movie_id,
                    'movie_title': new_rec['movie']['title'],
                    'old_rating': round(old_rating, 2),
                    'new_rating': round(new_rating, 2),
                    'change': round(new_rating - old_rating, 2)
                })
        
        # Generate summary
        change_summary = []
        if new_movies:
            change_summary.append(f"{len(new_movies)} new movies added")
        if removed_movies:
            change_summary.append(f"{len(removed_movies)} movies removed")
        if position_changes:
            change_summary.append(f"{len(position_changes)} movies changed position")
        if rating_changes:
            change_summary.append(f"{len(rating_changes)} movies had rating changes")
        
        return {
            'new_movies': len(new_movies),
            'removed_movies': len(removed_movies),
            'position_changes': position_changes[:10],  # Limit to top 10
            'rating_changes': rating_changes[:10],  # Limit to top 10
            'change_summary': '; '.join(change_summary) if change_summary else 'No significant changes'
        }
    
    def cleanup_expired_data(self):
        """Clean up expired temporary data."""
        current_time = time.time()
        
        # Clean session cache (5 minute TTL)
        expired_sessions = [
            session_id for session_id, data in self.session_cache.items()
            if current_time - data['timestamp'] > 300
        ]
        
        for session_id in expired_sessions:
            del self.session_cache[session_id]
        
        # Clean temporary storage (1 hour TTL)
        expired_temp = [
            key for key, data in self.temp_storage.items()
            if current_time - data.get('timestamp', 0) > 3600
        ]
        
        for key in expired_temp:
            del self.temp_storage[key]
        
        if expired_sessions or expired_temp:
            logger.info(f"Cleaned up {len(expired_sessions)} session cache entries and {len(expired_temp)} temp storage entries")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session management statistics."""
        return {
            'active_sessions': len(self.session_cache),
            'temp_storage_items': len(self.temp_storage),
            'suspicious_sessions': len(self.validator.suspicious_sessions),
            'analytics_summary': self.analytics.get_analytics_summary()
        }


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get or create session manager instance."""
    global _session_manager
    
    if _session_manager is None:
        _session_manager = SessionManager()
    
    return _session_manager