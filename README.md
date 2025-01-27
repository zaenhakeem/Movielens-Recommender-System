# Movielens-Recommender-System

# Collaborators
- Eugene Asengi
- Lilian Baburo
- Abdihakim Issack
- Samuel Yashue
- Brian Siele
# Overview

In today's digital age, personalized recommendations are critical to enhancing user experiences across various platforms. One prime example is the movie industry, where vast catalogs of films can overwhelm users, making it challenging for them to find content that suits their unique tastes. The MovieLens Recommender System project aims to address this challenge by leveraging the MovieLens dataset, a rich source of user ratings and movie metadata, to develop a robust recommendation engine.

Through this project, we aim to demonstrate the potential of data-driven recommendations in creating tailored viewing experiences, helping users discover movies they'll love, and fostering greater engagement with the platform.

---
## Table of Contents
- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
  - [k-Nearest Neighbors (k-NN)](#k-nearest-neighbors-k-nn)
  - [Evaluation](#evaluation)
- [Movie Metadata Scraper](#movie-metadata-scraper)
  - [Web Scraping for Movie Metadata (Images and URLs)](#web-scraping-for-movie-metadata-images-and-urls)
- [Conclusion](#conclusion)
 
---
## Data Collection and Preprocessing
### Dataset
The dataset contains the following key features:
- **Links.csv**: `vieId`, `imdbId`, `tmdbId`
- **Movies.cvs**:`movieId`,`title`,`genres`
- **ratings.cvs**:`userId`,`movieId`,`rating`,`timestamp`
- **Tags.cvs**:`userId`,`movieId`,`tag`,`timestamp`

## Data Preprocessing
The first step in building the recommender system is data preprocessing.
### 1. Cleaning the 'genres' Column:
The **'genres'** column contains a string of genres separated by a pipe (`|`). For easier manipulation and analysis, we split this column into a list of genres for each movie. This will allow us to:

- Efficiently analyze individual genres.
- Explore how genres relate to other movie attributes such as ratings, popularity, or release year.

By transforming this column, we can also conduct more targeted analysis of specific genres in the dataset.

### 2. Extracting the Year from the 'title' Column:
The **'title'** column contains the movie title with the year of release in parentheses at the end (e.g., "The Matrix (1999)"). To facilitate year-based analysis and comparisons, we extract the **year of release** from the title and store it in a new column called **'year'**.

This step enables us to:

- Perform time-based analysis, such as sorting movies by their release year.
- Analyze trends over time, such as how the popularity of certain genres evolves.
- Cleanly separate movie titles from their release years for better readability.
### Sparse User-Item Interaction Matrix: 
In this step, we create a **Sparse User-Item Interaction Matrix** to represent interactions between users and movies. This matrix is used to train recommendation models. Since most users only rate a small number of movies, the matrix is sparse, meaning most of its values are zeros.

We use the **Compressed Sparse Row (CSR)** format to store this matrix efficiently, as it saves memory and computational resources by only keeping track of non-zero ratings.

Additionally, several **mappings** are generated to help us access user and movie information:

- `user_mapper`: Maps user ID to user index.
- `movie_mapper`: Maps movie ID to movie index.
- `user_inv_mapper`: Maps user index back to user ID.
- `movie_inv_mapper`: Maps movie index back to movie ID.

These mappings are crucial for efficient matrix-based operations in building the recommendation system.

## Exploratory Data Analysis (EDA)
In this part, we look at the main details of the dataset to understand it better:
1. **Total Ratings:**  
   The dataset has **100,836 ratings** in total. This shows how many times users have rated movies.
2. **Unique Movies:**  
   There are **9,742 different movies** in the dataset. This tells us how many distinct movies have been rated by users.
3. **Unique Users:**  
   **610 users** have given ratings. This is the number of people who have rated at least one movie.
4. **Average Ratings per User:**  
   On average, each user has rated **165.3 movies**. This means that users are quite active, but some may have rated many more movies than others.
5. **Average Ratings per Movie:**  
   Each movie has been rated **10.35 times** on average. This means some movies are rated by many users, while others may have very few ratings.

### Key Insights:
- The dataset has a large number of ratings, which gives us a lot of data to work with.
- There are many movies, but not all movies have been rated by all users.
- Users seem active, but some rate more movies than others.
- Not all movies get the same amount of attention; some are rated much more often than others.

These insights help us understand the dataset and guide our analysis.


## Modelling
In this project, several machine learning algorithms were applied to build a movie recommendation system. Two key algorithms were implemented for this task: k-Nearest Neighbors (k-NN) and Singular Value Decomposition (SVD).
### k-Nearest Neighbors (k-NN)

In this step, we implement **item-item recommendations** using **k-Nearest Neighbors (k-NN)**. This technique recommends movies that are similar to the ones a user has interacted with, based on their engagement with various movies.

### Process:
1. **Input:** A function is created that takes a **movie_id** and the **user-item interaction matrix (X)** as input.
2. **Similarity Calculation:** We calculate the **cosine similarity** between movies (alternatively, we can use other distance measures like Euclidean or Manhattan distance) to measure how similar they are to each other.
3. **Top k Similar Movies:** The function then returns the top **k most similar movies** based on the similarity scores.

This method helps recommend movies that are **similar** to a user's past interactions, enabling personalized suggestions for the user based on their previous ratings or choices.

### Evaluation
The model's performance was evaluated using various metrics:

- **Precision**: 0.1 (Only 10% of recommendations were relevant)
- **Recall**: 0.25 (25% of relevant items were recommended)
- **F1 Score**: 0.14 (Indicating low performance)
- **RMSE (Root Mean Squared Error)**: 0.8804  
- **SVD RMSE**: 0.8804  

#### **Possible Reasons for Low Performance:**
- **Cold-start problem**: Limited data or insufficient training.
- **Lack of relevant interactions**: Low overlap between user preferences and recommendations.
- **Limited diversity**: Biased recommendations affecting accuracy.
- **Sparse data**: Insufficient user-item interactions leading to poor predictions.


## Movie Metadata Scraper

This project involves scraping movie metadata, specifically URLs and images, from The Movie Database (TMDb) using a list of movie IDs (TMDb IDs). The scraper generates URLs for each movie and extracts the associated backdrop image URL.

### Web Scraping for Movie Metadata (Images and URLs)

The web scraping process works as follows:

1. **Generate Movie URL**: For each movie, the scraper constructs a URL based on its unique TMDb ID. The format of the URL is:

2. **Send HTTP Request**: A GET request is made to fetch the webpage content for each movie.

3. **Parse HTML**: The HTML content of the response is parsed using BeautifulSoup to navigate and extract relevant data.

4. **Extract Image URL**: The scraper searches for the movie's backdrop image within the HTML structure and retrieves its URL. The image URL is then formatted to be a full URL:

5. **Handle Missing Images**: If no image is found for a particular movie, a default placeholder image URL is used:
6. **Store Data**: The URL and image URL for each movie are saved in a pandas DataFrame for further processing or storage.

## Conclusion
