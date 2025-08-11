#!/usr/bin/env python3
"""
Demonstration script for session management and rating updates functionality.
This script shows how the session management system works with rating updates.
"""

import asyncio
import json
from typing import Dict, Any

from app.services.session_manager import get_session_manager
from app.models.recommendation import UpdateRecommendationRequest


async def demo_session_management():
    """Demonstrate session management functionality."""
    print("🎬 Movie Recommendation System - Session Management Demo")
    print("=" * 60)
    
    # Get session manager
    session_manager = get_session_manager()
    
    # Demo 1: Session Validation
    print("\n1. Session Validation Demo")
    print("-" * 30)
    
    # Valid session data
    valid_session = {
        'ratings': {
            1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5,
            6: 3.0, 7: 4.0, 8: 3.5, 9: 5.0, 10: 2.5,
            11: 4.0, 12: 3.5, 13: 5.0, 14: 2.0, 15: 4.5
        }
    }
    
    is_valid, error_msg = session_manager.validate_session(valid_session)
    print(f"Valid session: {is_valid}")
    if error_msg:
        print(f"Error: {error_msg}")
    
    # Invalid session data (suspicious pattern)
    suspicious_session = {
        'ratings': {i: 4.0 for i in range(1, 21)}  # All identical ratings
    }
    
    is_valid, error_msg = session_manager.validate_session(suspicious_session)
    print(f"Suspicious session: {is_valid}")
    if error_msg:
        print(f"Error: {error_msg}")
    
    # Demo 2: Session Analytics
    print("\n2. Session Analytics Demo")
    print("-" * 30)
    
    ratings = {i: 3.5 + (i % 4) * 0.5 for i in range(1, 21)}
    analysis = session_manager.analyze_session(ratings, 'hybrid')
    
    print(f"Session ID: {analysis['session_id']}")
    print(f"Total Ratings: {analysis['total_ratings']}")
    print(f"Confidence Score: {analysis['confidence_score']:.2f}")
    print(f"Algorithm Used: {analysis['algorithm_used']}")
    
    pattern = analysis['pattern_analysis']
    print(f"Mean Rating: {pattern['mean_rating']}")
    print(f"Rating Variance: {pattern['rating_variance']}")
    print(f"Is Generous Rater: {pattern['is_generous_rater']}")
    
    # Demo 3: Session Optimization
    print("\n3. Session Data Optimization Demo")
    print("-" * 30)
    
    unordered_ratings = {5: 4.0, 1: 3.5, 3: 5.0, 2: 2.0}
    optimizations = session_manager.optimize_session_data(unordered_ratings)
    
    print(f"Original ratings: {unordered_ratings}")
    print(f"Sorted ratings: {optimizations['sorted_ratings']}")
    print(f"Mean rating: {optimizations['mean_rating']:.2f}")
    print(f"Rating count: {optimizations['rating_count']}")
    
    # Demo 4: Recommendation Delta Calculation
    print("\n4. Recommendation Delta Calculation Demo")
    print("-" * 30)
    
    old_recommendations = [
        {'movie': {'id': 1, 'title': 'The Shawshank Redemption'}, 'predicted_rating': 4.5},
        {'movie': {'id': 2, 'title': 'The Godfather'}, 'predicted_rating': 4.3},
        {'movie': {'id': 3, 'title': 'Pulp Fiction'}, 'predicted_rating': 4.1}
    ]
    
    new_recommendations = [
        {'movie': {'id': 1, 'title': 'The Shawshank Redemption'}, 'predicted_rating': 4.7},  # Rating increased
        {'movie': {'id': 3, 'title': 'Pulp Fiction'}, 'predicted_rating': 4.1},  # Same
        {'movie': {'id': 4, 'title': 'The Dark Knight'}, 'predicted_rating': 4.6}  # New movie
    ]
    
    delta = session_manager.calculate_recommendation_delta(old_recommendations, new_recommendations)
    
    print(f"New movies: {delta['new_movies']}")
    print(f"Removed movies: {delta['removed_movies']}")
    print(f"Rating changes: {len(delta['rating_changes'])}")
    print(f"Change summary: {delta['change_summary']}")
    
    if delta['rating_changes']:
        print("Rating changes:")
        for change in delta['rating_changes']:
            print(f"  - {change['movie_title']}: {change['old_rating']} → {change['new_rating']}")
    
    # Demo 5: Session Statistics
    print("\n5. Session Statistics Demo")
    print("-" * 30)
    
    stats = session_manager.get_session_stats()
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Temp storage items: {stats['temp_storage_items']}")
    print(f"Suspicious sessions: {stats['suspicious_sessions']}")
    
    analytics_summary = stats['analytics_summary']
    print(f"Total sessions analyzed: {analytics_summary['total_sessions']}")
    
    # Demo 6: Rate Limiting
    print("\n6. Rate Limiting Demo")
    print("-" * 30)
    
    session_id = session_manager.generate_session_id(ratings)
    print(f"Generated session ID: {session_id}")
    
    # Simulate multiple requests
    for i in range(3):
        allowed = session_manager.validator.check_rate_limit(session_id, max_requests=5, window_minutes=60)
        print(f"Request {i+1}: {'Allowed' if allowed else 'Rate limited'}")
    
    # Demo 7: Cleanup
    print("\n7. Cleanup Demo")
    print("-" * 30)
    
    print("Cleaning up expired data...")
    session_manager.cleanup_expired_data()
    print("Cleanup completed!")
    
    print("\n" + "=" * 60)
    print("✅ Session Management Demo Completed Successfully!")
    print("All session management features are working correctly.")


def demo_update_request_validation():
    """Demonstrate update request validation."""
    print("\n🔄 Update Request Validation Demo")
    print("-" * 40)
    
    # Valid update request
    try:
        valid_request = UpdateRecommendationRequest(
            existing_ratings={1: 4.0, 2: 3.5, 3: 5.0, 4: 2.0, 5: 4.5},
            new_ratings={6: 4.0, 7: 3.0},
            algorithm="hybrid",
            num_recommendations=20
        )
        print("✅ Valid update request created successfully")
        print(f"   Existing ratings: {len(valid_request.existing_ratings)}")
        print(f"   New ratings: {len(valid_request.new_ratings)}")
        print(f"   Algorithm: {valid_request.algorithm}")
    except Exception as e:
        print(f"❌ Error creating valid request: {e}")
    
    # Invalid update request (invalid rating values)
    try:
        invalid_request = UpdateRecommendationRequest(
            existing_ratings={1: 6.0},  # Invalid rating > 5.0
            new_ratings={2: 4.0},
            algorithm="hybrid"
        )
        print("❌ Invalid request should have failed validation")
    except Exception as e:
        print(f"✅ Invalid request correctly rejected: {e}")


if __name__ == "__main__":
    print("Starting Session Management Demo...")
    asyncio.run(demo_session_management())
    demo_update_request_validation()