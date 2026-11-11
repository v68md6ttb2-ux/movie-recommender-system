import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

print("Loading PyTorch Neural Collaborative Filtering model...")
try:
    # Define the model class
    class NeuralCollaborativeFiltering(nn.Module):
        def __init__(self, num_users: int, num_movies: int,
                     embedding_dim: int = 50):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

            self.fc_layers = nn.Sequential(
                nn.Linear(embedding_dim * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, user_ids, movie_ids):
            user_vec = self.user_embedding(user_ids)
            movie_vec = self.movie_embedding(movie_ids)
            x = torch.cat([user_vec, movie_vec], dim=1)
            return self.fc_layers(x).squeeze()
    
    # Load index mappings
    with open("pytorch/models/user_id_to_index.json", 'r') as f:
        user_to_index = json.load(f)
    
    with open("pytorch/models/movie_id_to_index.json", 'r') as f:
        movie_to_index = json.load(f)
    
    # Load movie data
    movies_df = pd.read_csv("data/ml-latest-small/movies.csv")
    movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    idx2movie = {v: k for k, v in movie_to_index.items()}
    
    # Model parameters
    num_users = len(user_to_index)
    num_movies = len(movie_to_index)
    
    # Load the model
    model = NeuralCollaborativeFiltering(num_users, num_movies)
    model.load_state_dict(torch.load("pytorch/models/neural_cf_model.pth", map_location=torch.device('cpu')))
    model.eval()
    
    print("Model loaded successfully!")
    
    # Display model information
    print(f"\nModel statistics:")
    print(f"- Number of users: {num_users}")
    print(f"- Number of movies: {num_movies}")
    print(f"- Embedding dimension: 50")
    print(f"- Neural network layers: 3-layer fully connected network")
    
    # Function to generate recommendations
    def generate_recommendations(user_id, top_n=5):
        try:
            # Input validation
            if not isinstance(user_id, (int, str)):
                raise ValueError("User ID must be an integer or string")
            
            if not isinstance(top_n, int) or top_n <= 0:
                raise ValueError("Number of recommendations must be a positive integer")
            
            # User ID validation
            user_id_str = str(user_id)
            if user_id_str not in user_to_index:
                return None
            
            user_idx = int(user_to_index[user_id_str])
            user_indices = torch.tensor([user_idx] * num_movies, dtype=torch.long)
            movie_indices = torch.tensor(list(range(num_movies)), dtype=torch.long)
            
            with torch.no_grad():
                scores = model(user_indices, movie_indices).numpy()
            
            top_indices = np.argsort(scores)[::-1][:top_n]
            
            recommendations = []
            for idx in top_indices:
                movie_id = idx2movie[idx]
                recommendations.append({
                    "movie_id": int(movie_id),
                    "title": movieid_to_title.get(int(movie_id), "Unknown Title"),
                    "predicted_rating": float(scores[idx])
                })
            
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return None
    
    # Test several user IDs
    test_user_ids = [1, 5, 10]
    print("\nGenerating user recommendations:")
    for user_id in test_user_ids:
        recs = generate_recommendations(user_id)
        if recs:
            print(f"\nRecommendations for user {user_id} (Neural Collaborative Filtering):")
            for i, rec in enumerate(recs, 1):
                try:
                    print(f"{i}. {rec['title']} (ID: {rec['movie_id']}, Predicted Rating: {rec['predicted_rating']:.2f})")
                except UnicodeEncodeError:
                    # Handle encoding issues by replacing problematic characters
                    title = rec['title'].encode('ascii', 'replace').decode('ascii')
                    print(f"{i}. {title} (ID: {rec['movie_id']}, Predicted Rating: {rec['predicted_rating']:.2f})")
        else:
            print(f"User {user_id} does not exist in the model")
    
    # Analyze model parameters
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    fc_params = sum(p.numel() for n, p in model.named_parameters() if 'fc' in n)
    
    print(f"\nModel parameter statistics:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Embedding layer parameters: {embedding_params:,} ({embedding_params/total_params*100:.1f}%)")
    print(f"- Fully connected layer parameters: {fc_params:,} ({fc_params/total_params*100:.1f}%)")
    
except Exception as e:
    print(f"Error loading or testing the model: {e}")
    import traceback
    traceback.print_exc()