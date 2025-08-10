import pickle
import json
import numpy as np

print("Testing model loading...")

# Test collaborative models
try:
    with open('data/models/collaborative_svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    print(f"SVD Model loaded: {type(svd_model)}")
    print(f"SVD components: {svd_model.n_components}")
except Exception as e:
    print(f"SVD model error: {e}")

try:
    with open('data/models/user_similarity_matrix.pkl', 'rb') as f:
        user_sim = pickle.load(f)
    print(f"User similarity shape: {user_sim.shape}")
except Exception as e:
    print(f"User similarity error: {e}")

try:
    with open('data/models/item_similarity_matrix.pkl', 'rb') as f:
        item_sim = pickle.load(f)
    print(f"Item similarity shape: {item_sim.shape}")
except Exception as e:
    print(f"Item similarity error: {e}")

# Test content models
try:
    with open('data/models/content_similarity_matrix.pkl', 'rb') as f:
        content_sim = pickle.load(f)
    print(f"Content similarity shape: {content_sim.shape}")
except Exception as e:
    print(f"Content similarity error: {e}")

# Test metrics
try:
    with open('data/models/collaborative_metrics.json', 'r') as f:
        collab_metrics = json.load(f)
    print(f"Collaborative RMSE: {collab_metrics.get('item_based_rmse', 'N/A')}")
except Exception as e:
    print(f"Collaborative metrics error: {e}")

try:
    with open('data/models/content_metrics.json', 'r') as f:
        content_metrics = json.load(f)
    print(f"Content RMSE: {content_metrics.get('content_rmse', 'N/A')}")
except Exception as e:
    print(f"Content metrics error: {e}")

print("Model testing complete!")