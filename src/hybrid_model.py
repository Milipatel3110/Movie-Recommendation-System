from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, array
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler
from pyspark.ml.linalg import Vectors
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("HybridRecommender").getOrCreate()

# Load movie metadata and ratings data
movies_metadata_path = "hdfs:///user/data/cleaned_movies_metadata.csv"
ratings_path = "hdfs:///user/data/cleaned_ratings.csv"
movies_df = spark.read.csv(movies_metadata_path, header=True, inferSchema=True)
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# Content-Based Filtering Part
# Tokenize and apply TF-IDF on overview
tokenizer = Tokenizer(inputCol="overview", outputCol="overview_tokens")
movies_df = tokenizer.transform(movies_df)

hashing_tf = HashingTF(inputCol="overview_tokens", outputCol="overview_tf", numFeatures=1000)
movies_df = hashing_tf.transform(movies_df)

idf = IDF(inputCol="overview_tf", outputCol="overview_tfidf")
idf_model = idf.fit(movies_df)
movies_df = idf_model.transform(movies_df)

# Vector assembler for content features
assembler = VectorAssembler(inputCols=["overview_tfidf"], outputCol="content_features")
movies_df = assembler.transform(movies_df)

# Collect content features for KNN
content_data = movies_df.select("movie_id", "title", "content_features").collect()
movie_ids = [row["movie_id"] for row in content_data]
movie_titles = [row["title"] for row in content_data]
content_vectors = np.array([row["content_features"].toArray() for row in content_data])

# KNN model for content-based recommendations
knn = NearestNeighbors(n_neighbors=10, metric='cosine')
knn.fit(content_vectors)

def get_similar_movies_content(movie_title, top_n=10):
    try:
        movie_index = movie_titles.index(movie_title)
    except ValueError:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return []
    
    target_vector = content_vectors[movie_index].reshape(1, -1)
    distances, indices = knn.kneighbors(target_vector, n_neighbors=top_n+1)
    
    similar_movies = []
    for i, idx in enumerate(indices.flatten()):
        if i == 0:
            continue
        similar_movies.append((movie_ids[idx], 1 - distances.flatten()[i]))
    
    return similar_movies

# Collaborative Filtering Part - User-Based
user_avg_ratings = ratings_df.groupBy("userId").avg("rating").withColumnRenamed("avg(rating)", "user_avg_rating")
ratings_with_avg = ratings_df.join(user_avg_ratings, on="userId")
ratings_df = ratings_with_avg.withColumn("rating_norm", col("rating") - col("user_avg_rating"))

# Pivot ratings to create a user-item matrix
user_item_matrix = ratings_df.groupBy("userId").pivot("movieId").avg("rating_norm").na.fill(0)

# Convert to RDD for similarity calculation
user_item_rdd = user_item_matrix.rdd.map(lambda row: (row[0], np.array(row[1:])))

# Hybrid Recommendation Function
def hybrid_recommendation(user_id, movie_title, top_n=10):
    similar_content_movies = get_similar_movies_content(movie_title, top_n=top_n)
    
    # Collaborative filtering scores for similar movies
    user_ratings = user_item_rdd.filter(lambda x: x[0] == user_id).collect()
    if not user_ratings:
        print(f"User {user_id} not found in the dataset.")
        return []
    
    user_ratings_vector = user_ratings[0][1]
    
    hybrid_scores = []
    for movie_id, content_similarity in similar_content_movies:
        movie_index = movie_ids.index(movie_id)
        user_rating_for_movie = user_ratings_vector[movie_index]
        hybrid_score = content_similarity * 0.5 + user_rating_for_movie * 0.5
        hybrid_scores.append((movie_id, hybrid_score))
    
    # Sort and return top recommendations
    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
    top_recommendations = [(movie_titles[movie_ids.index(movie)], score) for movie, score in hybrid_scores[:top_n]]
    
    return top_recommendations

# Example Usage
user_id = 1
movie_title = "The Matrix"
top_n = 10
recommendations = hybrid_recommendation(user_id, movie_title, top_n=top_n)

print(f"Top {top_n} recommendations for user {user_id} based on '{movie_title}':")
for idx, (title, score) in enumerate(recommendations, 1):
    print(f"{idx}. {title} (Hybrid Score: {score:.2f})")

# Stop Spark session
spark.stop()
