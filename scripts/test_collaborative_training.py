#!/usr/bin/env python3
"""
Test script for collaborative filtering training to ensure it works correctly
before running in Google Colab.

This script performs basic validation of the training pipeline with a small subset of data.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add the scripts directory to Python path
sys.path.append(str(Path(__file__).parent))

def test_data_loading():
    """Test if all required data files exist and can be loaded."""
    print("Testing data loading...")
    
    data_path = Path("data")
    
    # Check if files exist
    required_files = ["ratings.csv", "movies.json", "data_summary.json"]
    for file in required_files:
        file_path = data_path / file
        if not file_path.exists():
            print(f"❌ Missing required file: {file_path}")
            return False
        print(f"✅ Found: {file_path}")
    
    # Test loading ratings
    try:
        ratings_df = pd.read_csv(data_path / "ratings.csv")
        print(f"✅ Loaded ratings: {len(ratings_df)} records")
    except Exception as e:
        print(f"❌ Error loading ratings.csv: {e}")
        return False
    
    # Test loading movies
    try:
        with open(data_path / "movies.json", 'r') as f:
            movies_data = json.load(f)
        print(f"✅ Loaded movies: {len(movies_data)} records")
    except Exception as e:
        print(f"❌ Error loading movies.json: {e}")
        return False
    
    # Test loading data summary
    try:
        with open(data_path / "data_summary.json", 'r') as f:
            data_summary = json.load(f)
        print(f"✅ Loaded data summary with {len(data_summary)} keys")
    except Exception as e:
        print(f"❌ Error loading data_summary.json: {e}")
        return False
    
    return True

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        "pandas", "numpy", "sklearn", "matplotlib", "seaborn", "scipy"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            return False
    
    return True

def test_training_script_syntax():
    """Test if the training script has valid syntax."""
    print("Testing training script syntax...")
    
    script_path = Path("scripts/train_collaborative_filtering.py")
    
    if not script_path.exists():
        print(f"❌ Training script not found: {script_path}")
        return False
    
    try:
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # Try to compile the script
        compile(script_content, str(script_path), 'exec')
        print("✅ Training script syntax is valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in training script: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading training script: {e}")
        return False

def test_small_dataset():
    """Test training with a small subset of data."""
    print("Testing with small dataset...")
    
    try:
        # Import the trainer class
        from train_collaborative_filtering import CollaborativeFilteringTrainer
        
        # Create a trainer with modified parameters for testing
        trainer = CollaborativeFilteringTrainer()
        trainer.svd_components = 10  # Reduce for faster testing
        trainer.min_ratings_per_user = 5  # Lower threshold for testing
        trainer.min_ratings_per_movie = 3  # Lower threshold for testing
        
        # Load data
        trainer.load_data()
        print("✅ Data loaded successfully")
        
        # Take a small sample for testing
        sample_size = min(10000, len(trainer.ratings_df))
        trainer.ratings_df = trainer.ratings_df.sample(n=sample_size, random_state=42)
        print(f"✅ Using sample of {sample_size} ratings for testing")
        
        # Preprocess data
        trainer.preprocess_data()
        print("✅ Data preprocessing completed")
        
        # Create train/test split
        trainer.create_train_test_split()
        print("✅ Train/test split completed")
        
        # Test SVD training (quick test)
        trainer.train_svd_model()
        print("✅ SVD model training completed")
        
        # Test prediction function
        user_id = trainer.user_item_matrix.index[0]
        movie_id = trainer.user_item_matrix.columns[0]
        prediction = trainer.predict_svd(user_id, movie_id)
        print(f"✅ SVD prediction test: {prediction:.2f}")
        
        print("✅ Small dataset test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Small dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Collaborative Filtering Training Script")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Loading", test_data_loading),
        ("Script Syntax", test_training_script_syntax),
        ("Small Dataset Training", test_small_dataset)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 30)
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 50)
    print("🏁 TEST SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 All tests passed! The training script is ready for Google Colab.")
        print("\nNext steps:")
        print("1. Upload the script and data files to Google Colab")
        print("2. Run: exec(open('scripts/train_collaborative_filtering.py').read())")
        print("3. Monitor the training progress and results")
    else:
        print("⚠️  Some tests failed. Please fix the issues before running in Colab.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)