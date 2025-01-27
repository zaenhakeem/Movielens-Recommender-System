from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the saved KNN model and datasets
with open('knn_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

merged_df = pd.read_csv('merged_df.csv')
ratings_df = pd.read_csv('ratings.csv')

# Get unique user IDs
userIds = ratings_df['userId'].unique()

# Helper function to handle user-based recommendations
def recommend_for_user(user_id):
    """Generate movie recommendations for a given user ID."""
    try:
        user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
        unrated_movies = merged_df[~merged_df['movieId'].isin(user_movies)]
        k = 10  # Number of recommendations
        user_recommendations = loaded_model.get_neighbors(user_id, k=k)
        recommended_movies = unrated_movies[unrated_movies['movieId'].isin(user_recommendations)]
        return recommended_movies
    except ValueError:
        return pd.DataFrame()


# Helper function for content-based recommendations
def recommend_by_movie(movie_name):
    """Generate content-based recommendations based on a movie title."""
    result = process.extractOne(movie_name, merged_df['title'])
    closest_match = result[0]
    score = result[1]

    if closest_match is None or closest_match == "":
        return movie_name, pd.DataFrame()

    movie_name = closest_match
    numeric_features = merged_df.select_dtypes(include=[float, int]).copy()
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan).dropna()

    movie_vector = numeric_features[merged_df['title'] == movie_name].values
    movie_vector = np.nan_to_num(movie_vector, nan=0.0, posinf=0.0, neginf=0.0)

    if numeric_features.shape[1] != movie_vector.shape[1]:
        return movie_name, pd.DataFrame()

    similarities = cosine_similarity(numeric_features, movie_vector)
    similarities = similarities.flatten()

    if len(similarities) != len(numeric_features):
        return movie_name, pd.DataFrame()

    similarity_df = numeric_features.copy()
    similarity_df['similarity'] = similarities
    similarity_df['movieId'] = merged_df['movieId']
    similarity_df['title'] = merged_df['title']
    similarity_df['img_url'] = merged_df.get('img_url', 'https://via.placeholder.com/150')

    recommendations = similarity_df[similarity_df['title'] != movie_name] \
        .sort_values('similarity', ascending=False) \
        .head(10)[['title', 'similarity', 'img_url']]

    return movie_name, recommendations


@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html', user_ids=userIds)


@app.route('/recommend_user', methods=['POST'])
def recommend_for_user_route():
    """Handle user-based recommendations."""
    user_id = int(request.form['userId'])
    recommended_movies = recommend_for_user(user_id)

    if recommended_movies.empty:
        return render_template('recommendations.html', movies=[], message="User-Based Recommendations: No recommendations found.")
    else:
        movies = recommended_movies[['title', 'img_url']].to_dict('records')
        return render_template('recommendations.html', movies=movies, message="User-Based Recommendations")


@app.route('/recommend_movie', methods=['POST'])
def recommend_by_movie_route():
    """Handle content-based recommendations."""
    movie_name = request.form['movie_name']
    title, recommendations = recommend_by_movie(movie_name)

    if recommendations.empty:
        return render_template('recommendations.html', movies=[], message=f"Content-Based Recommendations: No recommendations found for '{movie_name}'")
    else:
        movies = recommendations.to_dict('records')
        return render_template('recommendations.html', movies=movies, message=f"Content-Based Recommendations for '{title}'")


if __name__ == '__main__':
    app.run(debug=True)
