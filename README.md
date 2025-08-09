# Movie Recommendation System

A modern, Netflix-inspired movie recommendation system that demonstrates end-to-end machine learning capabilities. This portfolio project showcases full-stack development skills with React frontend, FastAPI backend, and multiple ML recommendation algorithms.

## 🎯 Project Overview

This application provides personalized movie recommendations using collaborative filtering, content-based filtering, and hybrid approaches. Users can rate movies to receive tailored recommendations without requiring registration or authentication.

### Key Features

- **Multiple Recommendation Algorithms**: Collaborative filtering, content-based filtering, and hybrid approaches
- **Modern UI/UX**: Netflix-style dark theme with responsive design
- **Session-Based**: Privacy-first approach with no persistent user data
- **Real-Time Recommendations**: Fast inference using pre-trained models
- **Mobile-Responsive**: Optimized for desktop, tablet, and mobile devices

## 🛠 Technology Stack

### Frontend
- React 18 with Vite
- Tailwind CSS for styling
- Framer Motion for animations
- Axios for API communication

### Backend
- FastAPI for high-performance API
- Python 3.9+ with async/await
- Pydantic for data validation
- Scikit-learn for ML algorithms

### Machine Learning
- MovieLens 1M dataset
- Collaborative filtering (SVD, cosine similarity)
- Content-based filtering (TF-IDF)
- Hybrid recommendation system

## 📁 Project Structure

```
movie-recommender/
├── frontend/                 # React application
├── backend/                  # FastAPI application
├── data/                     # Processed datasets and models
├── notebooks/                # Jupyter notebooks for exploration
├── scripts/                  # Training scripts for Colab
├── docs/                     # Documentation and assets
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Node.js (v18+) and npm
- Python (v3.9+) and pip
- Git

### Installation

[Setup instructions will be added as development progresses]

## 📊 Dataset

This project uses the MovieLens 1M dataset, which contains:
- 1 million ratings from 6,000 users on 4,000 movies
- Movie metadata including titles, genres, and release years
- User demographic information

## 🤖 Machine Learning Approach

The recommendation system implements three main approaches:

1. **Collaborative Filtering**: Uses user-item interactions to find similar users and items
2. **Content-Based Filtering**: Recommends movies based on movie features and user preferences
3. **Hybrid System**: Combines both approaches for optimal recommendations

## 🎨 Design Philosophy

- **User Experience**: Intuitive, Netflix-inspired interface
- **Performance**: Fast loading and real-time recommendations
- **Privacy**: No user data persistence, session-based interactions
- **Accessibility**: Mobile-responsive and keyboard navigation support

## 📈 Development Status

This project is currently in development. Check the [tasks.md](.kiro/specs/movie-recommendation-system/tasks.md) file for detailed implementation progress.

## 🤝 Contributing

This is a portfolio project, but feedback and suggestions are welcome!

## 📄 License

This project is for educational and portfolio purposes.

---

**Note**: This README will be updated as development progresses with detailed setup instructions, API documentation, and deployment information.