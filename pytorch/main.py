from fastapi import FastAPI
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


app = FastAPI()


user_to_index = pd.read_json(
    "models/user_id_to_index.json", typ="series").to_dict()
movie_to_index = pd.read_json(
    "models/movie_id_to_index.json", typ="series").to_dict()

movies_df = pd.read_csv("../data/ml-latest-small/movies.csv")
movieid_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

idx2movie = {v: k for k, v in movie_to_index.items()}

num_users = len(user_to_index)
num_movies = len(movie_to_index)


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


# Load artifacts
model = NeuralCollaborativeFiltering(num_users, num_movies)
model.load_state_dict(torch.load("models/neural_cf_model.pth"))
model.eval()

all_movie_indices = torch.tensor(
    list(movie_to_index.values()), dtype=torch.long)


@app.get("/recommend/{user_id}")
def recommend(user_id: int, top_n: int = 5):
    if user_id not in user_to_index:
        return {"error": "Unknown user"}

    user_idx = torch.tensor([user_to_index[user_id]]
                            * num_movies, dtype=torch.long)

    with torch.inference_mode():
        scores = model(user_idx, all_movie_indices).numpy()

    top_indices = np.argsort(scores)[::-1][:top_n]

    recommendations = []
    for idx in top_indices:
        movie_id = int(idx2movie[idx])
        recommendations.append({
            "movie_id": movie_id,
            "title": movieid_to_title.get(movie_id, "Unknown Title"),
            "predicted_rating": float(scores[idx])
        })
    return {"user_id": user_id, "recommendations": recommendations}
