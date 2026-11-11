from flask import Flask, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and data
data = joblib.load("models/recommender_scikit_artifacts.pkl")

movies = data["movies_df"]
predicted_ratings = data["predicted_ratings"]

# For collaborative recommendation
user_to_idx = data["user_to_idx"]
movie_to_idx = data["movie_to_idx"]

# content based recommendation
movieid_to_idx = data["movieid_to_idx"]
tfidf_matrix = data["tfidf_matrix"]


def recommend_user_collab(user_id, top_n=10):
    if user_id not in user_to_idx:
        return []
    uid = user_to_idx[user_id]
    user_scores = predicted_ratings[uid].copy()
    top_idx = np.argsort(user_scores)[-top_n:][::-1]
    recommendations = movies.iloc[top_idx][["movieId", "title"]].to_dict(orient="records")
    result = {
        "user_id": user_id,
        "recommendations": recommendations
    }
    return result


def recommend_movie_content(movie_id, top_n=5):
    if movie_id not in movieid_to_idx:
        return []
    idx = movieid_to_idx[movie_id]
    sim_scores = list(enumerate((tfidf_matrix[idx] @ tfidf_matrix.T).toarray().ravel()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n+1]  # skip self
    movie_indices = [i for i, _ in sim_scores]
    recommendations = movies.iloc[movie_indices][["movieId", "title"]].to_dict(orient="records")
    return {"movie_id": movie_id, "recommendations": recommendations}


@app.route("/recommend/collab/<int:user_id>")
def rec_collab(user_id):
    return jsonify(recommend_user_collab(user_id, top_n=5))


@app.route("/recommend/content/<int:movie_id>")
def rec_content(movie_id):
    return jsonify(recommend_movie_content(movie_id, top_n=5))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
