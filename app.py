from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from fuzzywuzzy import process
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity function

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
        # Get movies watched by the user
        user_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
        
        # Get movies that the user has not rated
        unrated_movies = merged_df[~merged_df['movieId'].isin(user_movies)]
        
        k = 10  # Number of recommendations
        
        # Get the nearest neighbors for the user
        user_recommendations = loaded_model.get_neighbors(user_id, k=k)
        
        # Filter movies that are recommended and not rated by the user
        recommended_movies = unrated_movies[unrated_movies['movieId'].isin(user_recommendations)]
        
        return recommended_movies
    except ValueError:
        return pd.DataFrame()


# Helper function for content-based recommendations
def recommend_by_movie(movie_name):
    """Generate content-based recommendations based on a movie title."""
    
    # Use fuzzy matching to find the closest matching movie title
    result = process.extractOne(movie_name, merged_df['title'])
    closest_match = result[0]  # The movie title match
    score = result[1]          # The matching score

    # If no match is found, return an empty DataFrame
    if closest_match is None or closest_match == "":
        return movie_name, pd.DataFrame()

    # If a close match is found, update movie_name with the closest match
    movie_name = closest_match

    # Keep only numeric columns dynamically
    numeric_features = merged_df.select_dtypes(include=[float, int]).copy()

    # Replace infinities and drop NaNs
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Get the movie vector for the input movie
    movie_vector = numeric_features[merged_df['title'] == movie_name].values

    # Handle invalid values in movie_vector
    movie_vector = np.nan_to_num(movie_vector, nan=0.0, posinf=0.0, neginf=0.0)

    # Sanity check for matching dimensions
    if numeric_features.shape[1] != movie_vector.shape[1]:
        return movie_name, pd.DataFrame()  # Return empty DataFrame if dimensions mismatch

    # Calculate cosine similarity for all movies
    similarities = cosine_similarity(numeric_features, movie_vector)

    # Flatten the similarities to match the number of rows in numeric_features
    similarities = similarities.flatten()

    # Ensure the number of similarities matches the number of rows in numeric_features
    if len(similarities) != len(numeric_features):
        return movie_name, pd.DataFrame()  # Return empty if dimensions mismatch

    # Create a DataFrame with the similarity values and movie information
    similarity_df = numeric_features.copy()
    similarity_df['similarity'] = similarities

    # Merge with the movie title and movieId information from the original DataFrame
    similarity_df['movieId'] = merged_df['movieId']
    similarity_df['title'] = merged_df['title']

    # Exclude the input movie itself from the recommendations
    recommendations = similarity_df[similarity_df['title'] != movie_name] \
        .sort_values('similarity', ascending=False) \
        .head(10)[['title', 'similarity']]

    return movie_name, recommendations


# Route for homepage
@app.route('/')
def index():
    """Render the homepage."""
    return render_template('index.html', user_ids=userIds)


# Route for user-based recommendations
@app.route('/recommend_user', methods=['POST'])
def recommend_for_user_route():
    """Handle user-based recommendations."""
    user_id = int(request.form['userId'])
    recommended_movies = recommend_for_user(user_id)
    
    # If no recommendations found
    if recommended_movies.empty:
        return render_template('recommendations.html', movies=[], message="User-Based Recommendations: No recommendations found.")
    else:
        # Convert DataFrame to list of dictionaries
        movies = recommended_movies.to_dict('records')
        return render_template('recommendations.html', movies=movies, message="User-Based Recommendations")


# Route for content-based recommendations
@app.route('/recommend_movie', methods=['POST'])
def recommend_by_movie_route():
    """Handle content-based recommendations."""
    movie_name = request.form['movie_name']
    title, recommendations = recommend_by_movie(movie_name)
    
    # If no recommendations found
    if recommendations.empty:
        return render_template('recommendations.html', movies=[], message=f"Content-Based Recommendations: No recommendations found for '{movie_name}'")
    else:
        # Convert DataFrame to list of dictionaries
        movies = recommendations.to_dict('records')
        return render_template('recommendations.html', movies=movies, message=f"Content-Based Recommendations for '{title}'")


if __name__ == '__main__':
    app.run(debug=True)
