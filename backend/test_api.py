#!/usr/bin/env python3
"""
Simple API test script for the recommendation system.
"""

import asyncio
import json
from app.core.ml_models import ModelManager
from app.services.recommendation_engine import RecommendationEngine

async def test_recommendation_system():
    """Test the complete recommendation system."""
    print("Testing Movie Recommendation System...")
    
    # Initialize model manager
    model_manager = ModelManager()
    await model_manager.load_models()
    
    print(f"Models loaded: {model_manager.models_loaded}")
    print(f"System ready: {model_manager.is_ready()}")
    
    # Test model validation
    model_status = model_manager._validate_models()
    print(f"Model status: {model_status}")
    
    # Initialize recommendation engine
    engine = RecommendationEngine(model_manager)
    
    # Test with sample ratings
    sample_ratings = {i: 4.0 for i in range(1, 16)}  # 15 ratings of 4.0
    
    print(f"\nTesting with {len(sample_ratings)} sample ratings...")
    
    try:
        # Test hybrid recommendations (will fallback if models not available)
        response = await engine.generate_recommendations(
            ratings=sample_ratings,
            algorithm="hybrid",
            num_recommendations=5
        )
        
        print(f"\nRecommendation Results:")
        print(f"- Generated: {len(response.recommendations)} recommendations")
        print(f"- Algorithm used: {response.algorithm_used}")
        print(f"- Processing time: {response.processing_time:.3f}s")
        print(f"- Confidence score: {response.confidence_score:.2f}")
        print(f"- Total ratings: {response.total_ratings}")
        
        print(f"\nTop Recommendations:")
        for i, rec in enumerate(response.recommendations[:3], 1):
            print(f"{i}. {rec.movie.title}")
            print(f"   - Predicted rating: {rec.predicted_rating:.2f}")
            print(f"   - Confidence: {rec.confidence:.2f}")
            print(f"   - Explanation: {rec.explanation}")
            print(f"   - Genres: {', '.join(rec.movie.genres)}")
            print()
        
        # Test engine statistics
        stats = engine.get_engine_stats()
        print(f"Engine Statistics:")
        print(f"- Total requests: {stats['total_requests']}")
        print(f"- Average processing time: {stats['average_processing_time']:.3f}s")
        print(f"- Cache entries: {stats['cache_stats']['total_entries']}")
        
        print("\n✅ Recommendation system test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_recommendation_system())