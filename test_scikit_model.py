import joblib
import numpy as np
import pandas as pd

print("Loading Scikit-learn recommendation system model...")
try:
    # Load pre-trained model
    data = joblib.load("scikit_learn/models/recommender_scikit_artifacts.pkl")
    print("Model loaded successfully!")
    
    # Extract model components
    movies_df = data["movies_df"]
    predicted_ratings = data["predicted_ratings"]
    user_to_idx = data["user_to_idx"]
    movie_to_idx = data["movie_to_idx"]
    movieid_to_idx = data["movieid_to_idx"]
    tfidf_matrix = data["tfidf_matrix"]
    
    # Display model information
    print(f"\nModel statistics:")
    print(f"- Number of users: {len(user_to_idx)}")
    print(f"- Number of movies: {len(movie_to_idx)}")
    print(f"- Movie data shape: {movies_df.shape}")
    print(f"- Predicted ratings matrix shape: {predicted_ratings.shape}")
    print(f"- TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    # Generate recommendations for a specific user
    def recommend_user_collab(user_id, top_n=5):
        try:
            # Input validation
            if not isinstance(user_id, int):
                raise ValueError("User ID must be an integer")
            
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("Number of recommendations must be a positive integer")
            
            if user_id not in user_to_idx:
                return []
            
            uid = user_to_idx[user_id]
            user_scores = predicted_ratings[uid].copy()
            top_idx = np.argsort(user_scores)[-top_n:][::-1]
            recommendations = movies_df.iloc[top_idx][["movieId", "title"]].to_dict(orient="records")
            result = {
                "user_id": user_id,
                "recommendations": recommendations
            }
            return result
        except Exception as e:
            print(f"Error generating recommendations for user {user_id}: {str(e)}")
            return []
    
    # Test several user IDs
    test_user_ids = [1, 5, 10]
    print("\nGenerating user recommendations:")
    for user_id in test_user_ids:
        if user_id in user_to_idx:
            print(f"\nRecommendations for user {user_id}:")
            recs = recommend_user_collab(user_id)
            for i, rec in enumerate(recs["recommendations"], 1):
                print(f"{i}. {rec['title']} (ID: {rec['movieId']})")
        else:
            print(f"User {user_id} does not exist in the model")
    
    # Content-based recommendation example
    def recommend_movie_content(movie_id, top_n=3):
        try:
            # Input validation
            if not isinstance(movie_id, (int, str)):
                raise ValueError("Movie ID must be an integer or string")
            
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("Number of recommendations must be a positive integer")
            
            if movie_id not in movieid_to_idx:
                return []
            
            idx = movieid_to_idx[movie_id]
            sim_scores = list(
                enumerate((tfidf_matrix[idx] @ tfidf_matrix.T).toarray().ravel()))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]  # Skip self
            movie_indices = [i for i, _ in sim_scores]
            recommendations = movies_df.iloc[movie_indices][["movieId", "title"]].to_dict(orient="records")
            return recommendations
        except Exception as e:
            print(f"Error generating similar movie recommendations for movie {movie_id}: {str(e)}")
            return []
    
    # Test content-based recommendations
    print("\nContent-based recommendation examples:")
    sample_movies = movies_df.head(3)
    for _, movie in sample_movies.iterrows():
        print(f"\nMovies similar to '{movie['title']}':")
        similar_movies = recommend_movie_content(movie['movieId'])
        if similar_movies:
            for i, sim_movie in enumerate(similar_movies, 1):
                print(f"{i}. {sim_movie['title']} (ID: {sim_movie['movieId']})")
        else:
            print("No similar movies found")
    
except Exception as e:
    print(f"Error loading or testing the model: {e}")