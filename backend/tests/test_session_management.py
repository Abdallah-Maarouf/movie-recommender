"""
Tests for session management functionality.
"""

import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, Any

from app.services.session_manager import (
    SessionManager, 
    SessionAnalytics, 
    SessionValidator,
    get_session_manager
)


class TestSessionAnalytics:
    """Test session analytics functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analytics = SessionAnalytics()
    
    def test_analyze_rating_pattern_basic(self):
        """Test basic rating pattern analysis."""
        ratings = {1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5}
        
        pattern = self.analytics.analyze_rating_pattern(ratings)
        
        assert pattern['total_ratings'] == 5
        assert pattern['mean_rating'] == 3.8
        assert 'rating_distribution' in pattern
        assert 'rating_variance' in pattern
    
    def test_analyze_rating_pattern_empty(self):
        """Test rating pattern analysis with empty ratings."""
        pattern = self.analytics.analyze_rating_pattern({})
        assert pattern == {}
    
    def test_analyze_rating_pattern_generous_rater(self):
        """Test detection of generous raters."""
        ratings = {i: 4.5 for i in range(1, 21)}  # All high ratings
        
        pattern = self.analytics.analyze_rating_pattern(ratings)
        
        assert pattern['is_generous_rater'] == True
        assert pattern['mean_rating'] == 4.5
    
    def test_analyze_rating_pattern_critical_rater(self):
        """Test detection of critical raters."""
        ratings = {i: 2.0 for i in range(1, 21)}  # All low ratings
        
        pattern = self.analytics.analyze_rating_pattern(ratings)
        
        assert pattern['is_critical_rater'] == True
        assert pattern['mean_rating'] == 2.0
    
    def test_calculate_recommendation_confidence(self):
        """Test recommendation confidence calculation."""
        # Test with sufficient ratings
        ratings = {i: 3.5 + (i % 3) for i in range(1, 21)}
        confidence = self.analytics.calculate_recommendation_confidence(ratings, 'hybrid')
        
        assert 0.4 <= confidence <= 0.9
        
        # Test with insufficient ratings
        small_ratings = {1: 4.0, 2: 3.0}
        confidence = self.analytics.calculate_recommendation_confidence(small_ratings, 'hybrid')
        
        assert confidence == 0.3
    
    def test_track_session_metrics(self):
        """Test session metrics tracking."""
        session_id = "test_session_123"
        metrics = {
            'total_ratings': 20,
            'algorithm_used': 'hybrid',
            'confidence_score': 0.85
        }
        
        self.analytics.track_session_metrics(session_id, metrics)
        
        assert session_id in self.analytics.session_metrics
        assert self.analytics.session_metrics[session_id]['total_ratings'] == 20
        assert 'timestamp' in self.analytics.session_metrics[session_id]


class TestSessionValidator:
    """Test session validation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SessionValidator()
    
    def test_validate_session_data_valid(self):
        """Test validation of valid session data."""
        session_data = {
            'ratings': {
                '1': 4.0,
                '2': 3.5,
                '3': 5.0
            }
        }
        
        is_valid, error_msg = self.validator.validate_session_data(session_data)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_session_data_missing_ratings(self):
        """Test validation with missing ratings."""
        session_data = {}
        
        is_valid, error_msg = self.validator.validate_session_data(session_data)
        
        assert is_valid is False
        assert "Missing ratings data" in error_msg
    
    def test_validate_session_data_invalid_rating_values(self):
        """Test validation with invalid rating values."""
        session_data = {
            'ratings': {
                '1': 6.0,  # Invalid rating > 5.0
                '2': 3.5
            }
        }
        
        is_valid, error_msg = self.validator.validate_session_data(session_data)
        
        assert is_valid is False
        assert "Rating must be between 1.0 and 5.0" in error_msg
    
    def test_validate_session_data_invalid_movie_id(self):
        """Test validation with invalid movie IDs."""
        session_data = {
            'ratings': {
                '-1': 4.0,  # Invalid movie ID
                '2': 3.5
            }
        }
        
        is_valid, error_msg = self.validator.validate_session_data(session_data)
        
        assert is_valid is False
        assert "Invalid movie ID" in error_msg
    
    def test_detect_suspicious_patterns_identical_ratings(self):
        """Test detection of suspicious identical ratings."""
        # Create ratings with all identical values
        ratings = {i: 4.0 for i in range(1, 21)}
        
        is_suspicious = self.validator._detect_suspicious_patterns(ratings)
        
        assert is_suspicious is True
    
    def test_detect_suspicious_patterns_sequential_ids(self):
        """Test detection of suspicious sequential movie IDs."""
        # Create sequential movie IDs with identical ratings
        ratings = {i: 4.0 for i in range(1, 31)}
        
        is_suspicious = self.validator._detect_suspicious_patterns(ratings)
        
        assert is_suspicious is True
    
    def test_detect_suspicious_patterns_normal(self):
        """Test that normal patterns are not flagged as suspicious."""
        ratings = {
            1: 4.0, 5: 3.5, 10: 5.0, 15: 2.0, 20: 4.5,
            25: 3.0, 30: 4.0, 35: 3.5, 40: 5.0
        }
        
        is_suspicious = self.validator._detect_suspicious_patterns(ratings)
        
        assert is_suspicious is False
    
    def test_check_rate_limit(self):
        """Test rate limiting functionality."""
        session_id = "test_session"
        
        # Should allow requests within limit
        for i in range(5):
            result = self.validator.check_rate_limit(session_id, max_requests=10, window_minutes=60)
            assert result is True
        
        # Should block when limit exceeded
        for i in range(10):
            self.validator.check_rate_limit(session_id, max_requests=10, window_minutes=60)
        
        result = self.validator.check_rate_limit(session_id, max_requests=10, window_minutes=60)
        assert result is False
    
    def test_mark_and_check_suspicious(self):
        """Test marking and checking suspicious sessions."""
        session_id = "suspicious_session"
        
        assert self.validator.is_suspicious(session_id) is False
        
        self.validator.mark_suspicious(session_id)
        
        assert self.validator.is_suspicious(session_id) is True


class TestSessionManager:
    """Test session manager functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.session_manager = SessionManager()
    
    def test_generate_session_id(self):
        """Test session ID generation."""
        ratings1 = {1: 4.0, 2: 3.5}
        ratings2 = {2: 3.5, 1: 4.0}  # Same ratings, different order
        ratings3 = {1: 4.0, 2: 3.0}  # Different ratings
        
        id1 = self.session_manager.generate_session_id(ratings1)
        id2 = self.session_manager.generate_session_id(ratings2)
        id3 = self.session_manager.generate_session_id(ratings3)
        
        # Same ratings should generate same ID regardless of order
        assert id1 == id2
        # Different ratings should generate different IDs
        assert id1 != id3
        # IDs should be strings of expected length
        assert isinstance(id1, str)
        assert len(id1) == 16
    
    def test_validate_session_valid(self):
        """Test session validation with valid data."""
        session_data = {
            'ratings': {1: 4.0, 2: 3.5, 3: 5.0}
        }
        
        is_valid, error_msg = self.session_manager.validate_session(session_data)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_session_invalid(self):
        """Test session validation with invalid data."""
        session_data = {
            'ratings': {1: 6.0}  # Invalid rating
        }
        
        is_valid, error_msg = self.session_manager.validate_session(session_data)
        
        assert is_valid is False
        assert error_msg is not None
    
    def test_analyze_session(self):
        """Test session analysis."""
        ratings = {i: 3.5 + (i % 3) for i in range(1, 21)}
        
        analysis = self.session_manager.analyze_session(ratings, 'hybrid')
        
        assert 'session_id' in analysis
        assert 'total_ratings' in analysis
        assert 'confidence_score' in analysis
        assert 'pattern_analysis' in analysis
        assert analysis['total_ratings'] == 20
        assert analysis['algorithm_used'] == 'hybrid'
    
    def test_optimize_session_data(self):
        """Test session data optimization."""
        ratings = {3: 4.0, 1: 3.5, 2: 5.0}
        
        optimizations = self.session_manager.optimize_session_data(ratings)
        
        assert 'sorted_ratings' in optimizations
        assert 'rating_vector' in optimizations
        assert 'movie_ids' in optimizations
        assert 'mean_rating' in optimizations
        assert 'rating_count' in optimizations
        
        # Check that ratings are sorted
        sorted_ratings = optimizations['sorted_ratings']
        assert list(sorted_ratings.keys()) == [1, 2, 3]
    
    def test_calculate_recommendation_delta_initial(self):
        """Test recommendation delta calculation for initial recommendations."""
        new_recommendations = [
            {'movie': {'id': 1, 'title': 'Movie 1'}, 'predicted_rating': 4.0},
            {'movie': {'id': 2, 'title': 'Movie 2'}, 'predicted_rating': 3.5}
        ]
        
        delta = self.session_manager.calculate_recommendation_delta([], new_recommendations)
        
        assert delta['new_movies'] == 2
        assert delta['removed_movies'] == 0
        assert delta['change_summary'] == 'Initial recommendations generated'
    
    def test_calculate_recommendation_delta_updates(self):
        """Test recommendation delta calculation for updates."""
        old_recommendations = [
            {'movie': {'id': 1, 'title': 'Movie 1'}, 'predicted_rating': 4.0},
            {'movie': {'id': 2, 'title': 'Movie 2'}, 'predicted_rating': 3.5}
        ]
        
        new_recommendations = [
            {'movie': {'id': 1, 'title': 'Movie 1'}, 'predicted_rating': 4.2},  # Rating changed
            {'movie': {'id': 3, 'title': 'Movie 3'}, 'predicted_rating': 4.5}   # New movie
        ]
        
        delta = self.session_manager.calculate_recommendation_delta(old_recommendations, new_recommendations)
        
        assert delta['new_movies'] == 1  # Movie 3 is new
        assert delta['removed_movies'] == 1  # Movie 2 was removed
        assert len(delta['rating_changes']) > 0  # Movie 1 rating changed
    
    def test_cleanup_expired_data(self):
        """Test cleanup of expired data."""
        # Add some test data
        self.session_manager.session_cache['test1'] = {
            'optimizations': {},
            'timestamp': time.time() - 400  # Expired (> 300 seconds)
        }
        
        self.session_manager.session_cache['test2'] = {
            'optimizations': {},
            'timestamp': time.time() - 100  # Not expired
        }
        
        self.session_manager.temp_storage['temp1'] = {
            'data': {},
            'timestamp': time.time() - 4000  # Expired (> 3600 seconds)
        }
        
        # Run cleanup
        self.session_manager.cleanup_expired_data()
        
        # Check that expired data was removed
        assert 'test1' not in self.session_manager.session_cache
        assert 'test2' in self.session_manager.session_cache
        assert 'temp1' not in self.session_manager.temp_storage
    
    def test_get_session_stats(self):
        """Test session statistics retrieval."""
        # Add some test data
        self.session_manager.session_cache['test'] = {
            'optimizations': {},
            'timestamp': time.time()
        }
        
        stats = self.session_manager.get_session_stats()
        
        assert 'active_sessions' in stats
        assert 'temp_storage_items' in stats
        assert 'suspicious_sessions' in stats
        assert 'analytics_summary' in stats
        assert stats['active_sessions'] >= 1


class TestSessionManagerIntegration:
    """Integration tests for session manager."""
    
    def test_get_session_manager_singleton(self):
        """Test that get_session_manager returns singleton instance."""
        manager1 = get_session_manager()
        manager2 = get_session_manager()
        
        assert manager1 is manager2
        assert isinstance(manager1, SessionManager)
    
    @pytest.mark.asyncio
    async def test_full_session_workflow(self):
        """Test complete session workflow."""
        session_manager = get_session_manager()
        
        # Step 1: Validate session data
        session_data = {
            'ratings': {i: 3.5 + (i % 3) * 0.5 for i in range(1, 21)}  # More varied ratings
        }
        
        is_valid, error_msg = session_manager.validate_session(session_data)
        if not is_valid:
            print(f"Validation failed: {error_msg}")
        assert is_valid is True
        
        # Step 2: Analyze session
        analysis = session_manager.analyze_session(session_data['ratings'], 'hybrid')
        assert analysis['total_ratings'] == 20
        
        # Step 3: Optimize session data
        optimizations = session_manager.optimize_session_data(session_data['ratings'])
        assert 'sorted_ratings' in optimizations
        
        # Step 4: Get statistics
        stats = session_manager.get_session_stats()
        assert 'active_sessions' in stats


if __name__ == "__main__":
    pytest.main([__file__])