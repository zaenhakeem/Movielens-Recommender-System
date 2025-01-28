# **MovieLens Recommender System**

This project is a collaborative and content-based movie recommendation system utilizing the MovieLens dataset. It recommends movies to users based on their preferences and movie metadata.

---

## **Table of Contents**  
1. [Introduction](#introduction)  
2. [Dataset Overview](#dataset-overview)  
3. [Data Preprocessing](#data-preprocessing)  
4. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
5. [Collaborative Filtering Approach](#collaborative-filtering-approach)  
6. [Content-Based Recommendation Approach](#content-based-recommendation-approach)  
7. [Recommendation System Results](#recommendation-system-results)  
8. [Evaluation Metrics](#evaluation-metrics)  
9. [Deployment](#deployment)  
10. [Contributors](#contributors)  
11. [Conclusion](#conclusion)  

---

## **Introduction**  

Recommender systems play a critical role in providing personalized experiences in various domains like e-commerce and streaming platforms. This project aims to develop a recommendation system using the MovieLens dataset to:  
1. Suggest movies based on similar users' preferences (**Collaborative Filtering**).  
2. Suggest movies similar to a specific title (**Content-Based Filtering**).  

---  

## **Dataset Overview**  

The project uses the **MovieLens Dataset**, which includes:  
- **Movies Data**: Metadata about movies such as titles, genres, and IDs.  
- **Ratings Data**: User ratings for movies, including `userId`, `movieId`, and rating scores.  

### Key Characteristics:  
- **Number of Movies**: Approximately 9,000+  
- **Number of Users**: Over 600 unique user IDs  
- **Ratings**: Each user has rated several movies on a scale from 1 to 5  

---  

## **Data Preprocessing**  

### Key Steps:  
1. **Merging Datasets**: Combined movies and ratings datasets using `movieId`.  
2. **Handling Missing Values**: Removed rows with null values.  
3. **Filtering Rarely Rated Movies**: Excluded movies with very few ratings.  
4. **Encoding Data**: Processed categorical features (like genres) to numeric formats.  
5. **Normalizing Ratings**: Transformed rating distributions to ensure consistency.  

---  

## **Exploratory Data Analysis (EDA)**  

### Key Insights:  
- **Top-Rated Movies**: Movies with the highest average ratings.  
- **Most Rated Movies**: Analyzed movies frequently rated by users.  
- **User Activity**: Distribution of the number of ratings per user.  

Visualizations highlighted trends and user preferences, aiding in model design.  

---  

## **Collaborative Filtering Approach**  

This technique recommends movies based on user interactions with the system.  

### Process:  
1. **User-Item Matrix**: Created a matrix with users as rows, movies as columns, and ratings as values.  
2. **KNN Model**: Identified similar users using the K-Nearest Neighbors algorithm.  
3. **Recommendations**: Suggested movies that similar users highly rated but were unrated by the target user.  

---  

## **Content-Based Recommendation Approach**  

This method recommends movies based on similarity to a specific movie.  

### Process:  
1. **Feature Selection**: Focused on movie metadata like genres and titles.  
2. **Cosine Similarity**: Computed similarity between movie feature vectors.  
3. **Recommendations**: Suggested movies most similar to a selected movie.  

---  

## **Recommendation System Results**  

### Examples:  
1. **Collaborative Filtering**: Personalized results for users based on others with similar tastes.  
2. **Content-Based Filtering**: Relevant movies similar to a selected title, e.g., movies in the same genre.  

A Flask web app provides an intuitive, Netflix-style interface for showcasing recommendations.  

---  

## **Evaluation Metrics**  

Performance was evaluated using:  

1. **Precision**: Fraction of relevant movies in the recommendations.  
2. **Recall**: Fraction of relevant movies retrieved out of all relevant movies.  
3. **RMSE (Root Mean Square Error)**: Assessed accuracy of predicted ratings.  

These metrics demonstrated the system’s ability to generate relevant and personalized recommendations.  

---  

## **Deployment**  

The application is deployed online for testing and demonstration:  
[MovieLens Recommender System Deployment](https://movielens-recommender-system-rdyj.onrender.com/)  

---

## **Contributors**  

This project was developed by the following team members:  
1. **Abdihakim Issack**  
2. **Lilian Kaburo**  
3. **Eugene Asengi**  
4. **Samuel Yashua**  
5. **Brian Siele**
---  

## **Conclusion**  

The MovieLens Recommender System effectively combines collaborative filtering and content-based filtering techniques to deliver accurate and personalized movie recommendations. Future improvements could include:  
- Hybrid recommendation techniques.  
- Integration of additional metadata.  
- Optimization for scalability and real-time recommendations.  

---
