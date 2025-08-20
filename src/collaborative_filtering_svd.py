from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from scipy.sparse.linalg import svds
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("CollaborativeFilteringSVD").getOrCreate()

# Load ratings data
ratings_path = "hdfs:///user/data/cleaned_ratings.csv"
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# Convert Spark DataFrame to Pandas DataFrame for SVD processing
ratings_pd = ratings_df.select("userId", "movieId", "rating").toPandas()

# Create a user-item ratings matrix
user_item_matrix = ratings_pd.pivot(index="userId", columns="movieId", values="rating").fillna(0)

# De-mean the data (subtract mean of each user's ratings)
user_ratings_mean = user_item_matrix.mean(axis=1)
ratings_demeaned = user_item_matrix.sub(user_ratings_mean, axis=0)

# Convert to numpy array for SVD
ratings_demeaned_matrix = ratings_demeaned.values

# Perform SVD with k latent factors
U, sigma, Vt = svds(ratings_demeaned_matrix, k=50)
sigma = np.diag(sigma)

# Reconstruct the ratings matrix
predicted_ratings_matrix = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.values.reshape(-1, 1)

# Convert reconstructed matrix back to a DataFrame
predicted_ratings_df = pd.DataFrame(predicted_ratings_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

# Define a function to recommend movies for a specific user
def recommend_movies(predicted_ratings_df, user_id, original_ratings_df, top_n=10):
    # Get and sort the user's predicted ratings
    user_predictions = predicted_ratings_df.loc[user_id].sort_values(ascending=False)

    # Get movies the user has already rated
    user_rated_movies = original_ratings_df[original_ratings_df.userId == user_id]["movieId"].tolist()

    # Filter out movies the user has already rated
    recommendations = user_predictions[~user_predictions.index.isin(user_rated_movies)].head(top_n)
    
    return recommendations

# Example usage: Recommend top 10 movies for user 1
user_id = 1
recommendations = recommend_movies(predicted_ratings_df, user_id, ratings_pd, top_n=10)
print(f"Top 10 recommendations for User {user_id}:\n", recommendations)

# Stop Spark session
spark.stop()
