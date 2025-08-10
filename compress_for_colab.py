#!/usr/bin/env python3
"""
Compress files for efficient Colab upload
"""

import zipfile
import os
from pathlib import Path

def compress_for_colab():
    """Create a compressed archive with all necessary files"""
    
    # Files to include
    files_to_compress = [
        'scripts/train_hybrid_model.py',
        'data/ratings.csv',
        'data/movies.json',
        'data/models/collaborative_svd_model.pkl',
        'data/models/user_similarity_matrix.pkl',
        'data/models/item_similarity_matrix.pkl',
        'data/models/user_item_matrix.pkl',
        'data/models/content_similarity_matrix.pkl',
        'data/models/movie_features.pkl',
        'data/models/collaborative_metrics.json',
        'data/models/content_metrics.json'
    ]
    
    # Create compressed archive
    with zipfile.ZipFile('hybrid_model_colab.zip', 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zipf:
        for file_path in files_to_compress:
            if os.path.exists(file_path):
                zipf.write(file_path)
                print(f"✓ Added {file_path}")
            else:
                print(f"✗ Missing {file_path}")
    
    # Check final size
    size_mb = os.path.getsize('hybrid_model_colab.zip') / (1024 * 1024)
    print(f"\n📦 Created hybrid_model_colab.zip ({size_mb:.1f} MB)")
    print("Upload this single file to Colab!")

if __name__ == "__main__":
    compress_for_colab()