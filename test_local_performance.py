#!/usr/bin/env python3
"""
Quick performance test for local hybrid model execution
"""

import time
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

def test_local_performance():
    """Test how long key operations take locally"""
    
    print("🔍 Testing Local Performance...")
    print("=" * 50)
    
    data_path = Path("data")
    models_path = data_path / "models"
    
    # Test 1: Loading models
    start_time = time.time()
    try:
        with open(models_path / "collaborative_svd_model.pkl", 'rb') as f:
            collab_model = pickle.load(f)
        with open(models_path / "content_similarity_matrix.pkl", 'rb') as f:
            content_sim = pickle.load(f)
        with open(models_path / "item_similarity_matrix.pkl", 'rb') as f:
            item_sim = pickle.load(f)
        
        load_time = time.time() - start_time
        print(f"✓ Model Loading: {load_time:.2f} seconds")
        
    except Exception as e:
        print(f"✗ Model Loading Failed: {e}")
        return
    
    # Test 2: Loading data
    start_time = time.time()
    try:
        movies_df = pd.read_json(data_path / "movies.json")
        ratings_df = pd.read_csv(data_path / "ratings.csv")
        
        data_load_time = time.time() - start_time
        print(f"✓ Data Loading: {data_load_time:.2f} seconds")
        print(f"  - Movies: {len(movies_df):,}")
        print(f"  - Ratings: {len(ratings_df):,}")
        
    except Exception as e:
        print(f"✗ Data Loading Failed: {e}")
        return
    
    # Test 3: Sample similarity calculation
    start_time = time.time()
    try:
        # Test matrix operations
        sample_similarities = item_sim[:100, :100]  # Sample 100x100
        sample_calc = np.dot(sample_similarities, np.random.rand(100))
        
        calc_time = time.time() - start_time
        print(f"✓ Sample Calculations: {calc_time:.4f} seconds")
        
    except Exception as e:
        print(f"✗ Calculations Failed: {e}")
        return
    
    # Test 4: Memory usage estimate
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"✓ Current Memory Usage: {memory_mb:.1f} MB")
        
    except ImportError:
        print("ℹ️  Install psutil for memory monitoring: pip install psutil")
    
    # Estimate total time
    total_estimate = (load_time + data_load_time) * 10 + 300  # Conservative estimate
    print("\n" + "=" * 50)
    print(f"📊 ESTIMATED TOTAL RUNTIME: {total_estimate/60:.1f} minutes")
    print("=" * 50)
    
    if total_estimate < 900:  # Less than 15 minutes
        print("🚀 RECOMMENDATION: Run locally! It should be fast enough.")
    else:
        print("⚠️  RECOMMENDATION: Consider cloud execution for better performance.")
    
    return total_estimate

if __name__ == "__main__":
    test_local_performance()