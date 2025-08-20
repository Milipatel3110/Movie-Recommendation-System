from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, udf
from pyspark.ml.feature import CountVectorizer, IDF, Tokenizer, HashingTF
from pyspark.sql.types import FloatType
from pyspark.ml.linalg import Vectors
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("ContentBasedFiltering").getOrCreate()

# Load the main movies metadata
movies_metadata_path = "hdfs:///user/data/cleaned_movies_metadata.csv"
movies_df = spark.read.csv(movies_metadata_path, header=True, inferSchema=True)

# Load additional datasets: Cast and Keywords
cast_path = "hdfs:///user/data/cleaned_cast.csv"
keywords_path = "hdfs:///user/data/cleaned_keywords.csv"

cast_df = spark.read.csv(cast_path, header=True, inferSchema=True)
keywords_df = spark.read.csv(keywords_path, header=True, inferSchema=True)

# Select necessary columns and rename for consistency
movies_df = movies_df.select("id", "title", "genre_name", "overview", "release_year").withColumnRenamed("id", "movieId")
cast_df = cast_df.select("movie_id", "lead_actor").withColumnRenamed("movie_id", "movieId").withColumnRenamed("lead_actor", "cast")
keywords_df = keywords_df.select("movie_id", "keyword_name").withColumnRenamed("movie_id", "movieId").withColumnRenamed("keyword_name", "keywords")

# Join all data on `movieId`
movies_df = movies_df.join(cast_df, on="movieId", how="left")
movies_df = movies_df.join(keywords_df, on="movieId", how="left")

# Combine text features into a single field for vectorization
movies_df = movies_df.withColumn(
    "combined_features",
    concat_ws(" ", col("genre_name"), col("overview"), col("cast"), col("keywords"))
)

# Tokenize the combined text features
tokenizer = Tokenizer(inputCol="combined_features", outputCol="tokenized_features")
movies_df = tokenizer.transform(movies_df)

# HashingTF and IDF for feature extraction
hashing_tf = HashingTF(inputCol="tokenized_features", outputCol="raw_features", numFeatures=1000)
featurized_data = hashing_tf.transform(movies_df)

idf = IDF(inputCol="raw_features", outputCol="features")
idf_model = idf.fit(featurized_data)
tfidf_features = idf_model.transform(featurized_data)

# Select only necessary columns for similarity calculation
tfidf_features = tfidf_features.select("movieId", "title", "features")

# Function to calculate cosine similarity
def cosine_similarity(vec1, vec2):
    return float(vec1.dot(vec2) / (vec1.norm(2) * vec2.norm(2)))

# Register a UDF for calculating cosine similarity
@udf(FloatType())
def cosine_similarity_udf(features):
    if features is None:
        return float(0)
    return float(features.dot(broadcast_target_vector.value) / (features.norm(2) * broadcast_target_vector.value.norm(2)))

# Example of how to calculate similarity with a specific movie
def get_similar_movies(movie_title, top_n=10):
    # Get the feature vector of the target movie
    target_movie = tfidf_features.filter(tfidf_features.title == movie_title).select("features").first()

    if target_movie is None:
        print("Movie not found.")
        return

    # Broadcast the target vector to use in UDF
    global broadcast_target_vector
    broadcast_target_vector = spark.sparkContext.broadcast(target_movie.features)

    # Calculate similarity of all movies with the target movie using the UDF
    similarity_df = tfidf_features.withColumn("similarity", cosine_similarity_udf(col("features")))

    # Display top N similar movies
    similarity_df = similarity_df.orderBy(col("similarity").desc()).select("movieId", "title", "similarity")
    similarity_df.show(top_n, truncate=False)

    # Stop broadcasting after use
    broadcast_target_vector.unpersist()

# Example usage
get_similar_movies("The Matrix", top_n=10)

# Stop Spark session
spark.stop()
