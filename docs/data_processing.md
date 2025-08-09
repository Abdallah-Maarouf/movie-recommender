# MovieLens Data Processing Documentation

## Overview

This document describes the data processing pipeline for the MovieLens 1M dataset used in the movie recommendation system. The processing transforms raw MovieLens data into clean, structured formats suitable for both machine learning model training and web application use.

## Dataset Information

### Source Dataset: MovieLens 1M
- **Movies**: 3,883 movies with titles, genres, and release years
- **Ratings**: 1,000,209 ratings from 6,040 users on 3,706 movies
- **Users**: 6,040 users with demographic information
- **Rating Scale**: 1-5 stars
- **Time Period**: Collected over various time periods

### Data Quality Metrics
- **Movies with ratings**: 3,706 out of 3,883 movies (95.4%)
- **Movies with 100+ ratings**: 1,216 movies (31.3%)
- **Movies with 1000+ ratings**: 207 movies (5.3%)
- **Average ratings per movie**: 257.6 ratings
- **Median ratings per movie**: 32 ratings

## Data Processing Pipeline

### 1. Data Loading
The script loads three main data files from the `ml-1m/` directory:
- `movies.dat`: Movie metadata (ID, title, genres)
- `ratings.dat`: User ratings (UserID, MovieID, Rating, Timestamp)
- `users.dat`: User demographics (UserID, Gender, Age, Occupation, Zip-code)

### 2. Movie Data Cleaning

#### Title Processing
- **Year Extraction**: Extracts release year from movie titles using regex pattern `\((\d{4})\)$`
- **Clean Titles**: Removes year information from titles for display purposes
- **Web-Safe Titles**: Escapes special characters for HTML display (`"` → `&quot;`, `'` → `&#39;`)

#### Genre Processing
- **Genre Lists**: Splits pipe-separated genres into arrays
- **Genre Standardization**: Trims whitespace and handles missing values
- **Genre Distribution**: Tracks frequency of each genre across the dataset

#### Example Transformation
```
Original: "Toy Story (1995)" | "Animation|Children's|Comedy"
Processed: {
  "CleanTitle": "Toy Story",
  "WebTitle": "Toy Story",
  "Year": 1995,
  "GenreList": ["Animation", "Children's", "Comedy"]
}
```

### 3. Data Quality Analysis

#### Rating Statistics
For each movie, the system calculates:
- **Rating Count**: Total number of ratings received
- **Average Rating**: Mean rating score (1-5 scale)
- **Rating Standard Deviation**: Measure of rating consistency
- **Unique Users**: Number of distinct users who rated the movie

#### Quality Thresholds
- **Minimum Viable**: 100+ ratings (1,216 movies)
- **High Confidence**: 1000+ ratings (207 movies)
- **Popular Threshold**: Used for initial movie selection

### 4. Initial Movie Selection

#### Selection Criteria
The system selects 30 diverse movies for the initial rating interface using:

1. **Popularity Filter**: Movies with 1000+ ratings (high confidence)
2. **Genre Balance**: Representation across 8 major genres:
   - Action, Drama, Comedy, Thriller
   - Romance, Sci-Fi, Horror, Adventure
3. **Quality Ranking**: Highest-rated movies within each genre
4. **Fallback Strategy**: If insufficient movies meet criteria, threshold lowered to 500+ ratings

#### Selection Algorithm
```python
# Target 3-4 movies per major genre
movies_per_genre = max(1, target_count // len(target_genres))

# Fill remaining slots with highest-rated movies
# Ensure no duplicates across genre selections
```

### 5. Data Export Formats

#### movies.json (Full Dataset)
Complete movie metadata for all 3,883 movies:
```json
{
  "MovieID": 1,
  "Title": "Toy Story (1995)",
  "CleanTitle": "Toy Story",
  "WebTitle": "Toy Story",
  "Year": 1995,
  "GenreList": ["Animation", "Children's", "Comedy"],
  "rating_count": 2077,
  "avg_rating": 3.878
}
```

#### initial_movies.json (Curated Selection)
30 carefully selected movies for user rating interface:
- Balanced genre representation
- High rating confidence (1000+ ratings)
- Mix of popular and critically acclaimed films
- Optimized for user engagement

#### ratings.csv (Training Data)
Raw ratings matrix for machine learning:
```csv
UserID,MovieID,Rating,Timestamp
1,1193,5,978300760
1,661,3,978302109
```

#### data_summary.json (Quality Metrics)
Comprehensive dataset statistics:
- Movie and rating distributions
- Genre frequency analysis
- Data quality indicators
- Year range and coverage

## Data Validation

### Automated Checks
The processing script includes validation for:
- **File Existence**: Verifies all required input files are present
- **Data Integrity**: Checks for missing values and data type consistency
- **Export Verification**: Confirms all output files are created successfully
- **Count Validation**: Ensures record counts match expectations

### Quality Assurance
- **Genre Coverage**: Verifies balanced representation across major genres
- **Rating Thresholds**: Ensures selected movies meet minimum rating requirements
- **Character Encoding**: Handles special characters and international titles
- **JSON Validity**: Ensures all exported JSON files are properly formatted

## Usage in Application

### Frontend Integration
- `initial_movies.json`: Powers the initial movie rating interface
- Provides movie metadata for display (titles, years, genres)
- Enables genre-based filtering and sorting

### Backend Integration
- `movies.json`: Complete movie database for recommendations
- `ratings.csv`: Training data for machine learning models
- `data_summary.json`: System statistics and monitoring

### Model Training
- Ratings matrix supports collaborative filtering algorithms
- Movie metadata enables content-based filtering
- Genre information facilitates hybrid recommendation approaches

## Performance Considerations

### Processing Time
- **Data Loading**: ~2-3 seconds for full dataset
- **Cleaning Pipeline**: ~5-10 seconds for all transformations
- **Export Operations**: ~3-5 seconds for all output files
- **Total Runtime**: ~15-20 seconds end-to-end

### Memory Usage
- **Peak Memory**: ~200-300 MB during processing
- **Output Size**: ~15 MB total for all processed files
- **Scalability**: Pipeline designed for datasets up to 10M ratings

## Future Enhancements

### Potential Improvements
1. **External APIs**: Integration with TMDB for movie posters and additional metadata
2. **Real-time Updates**: Streaming data processing for new ratings
3. **Advanced Filtering**: Content-based features (directors, actors, keywords)
4. **Multilingual Support**: International title translations and genre mappings

### Monitoring and Maintenance
- **Data Freshness**: Regular updates with new MovieLens releases
- **Quality Monitoring**: Automated alerts for data quality degradation
- **Performance Tracking**: Processing time and memory usage monitoring

---

**Last Updated**: Generated during initial data processing
**Script Location**: `scripts/data_preprocessing.py`
**Output Directory**: `data/`