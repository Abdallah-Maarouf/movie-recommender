import pandas as pd
import json

# Check ratings data structure
print("Checking ratings data structure...")
ratings_df = pd.read_csv('data/ratings.csv')
print("Ratings columns:", ratings_df.columns.tolist())
print("Ratings shape:", ratings_df.shape)
print("First few rows:")
print(ratings_df.head())

# Check movies data structure
print("\nChecking movies data structure...")
with open('data/movies.json', 'r', encoding='utf-8') as f:
    movies_data = json.load(f)
movies_df = pd.DataFrame(movies_data)
print("Movies columns:", movies_df.columns.tolist())
print("Movies shape:", movies_df.shape)
print("First few rows:")
print(movies_df.head())