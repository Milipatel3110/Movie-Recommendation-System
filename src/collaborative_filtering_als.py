from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, expr, explode
from pyspark.ml.feature import StringIndexer

# Initialize Spark session
spark = SparkSession.builder.appName("CollaborativeFilteringALS").getOrCreate()

# Load the ratings data
ratings_path = "hdfs:///user/data/cleaned_ratings.csv"
ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)

# Load movies metadata
movies_metadata_path = "hdfs:///user/data/cleaned_movies_metadata.csv"
movies_df = spark.read.csv(movies_metadata_path, header=True, inferSchema=True)

# Select and rename columns
ratings_df = ratings_df.select("userId", "movieId", "rating")
movies_df = movies_df.select("id", "title", "genre_id", "genre_name", "release_year").withColumnRenamed("id", "movieId")

# Check for movieId mismatches between ratings_df and movies_df
missing_movie_ids = ratings_df.join(movies_df, on="movieId", how="left").filter(movies_df.movieId.isNull()).select("movieId").distinct()
missing_movie_ids_count = missing_movie_ids.count()
if missing_movie_ids_count > 0:
    print(f"Warning: {missing_movie_ids_count} movieIds in ratings_df are not found in movies_df.")

# Handle genre indexing
indexer = StringIndexer(inputCol="genre_name", outputCol="genres_indexed").fit(movies_df)
movies_df = indexer.transform(movies_df)

# Join ratings and movies data
enriched_ratings_df = ratings_df.join(movies_df, on="movieId", how="left")

# Fill NULL values in the metadata columns with default values
enriched_ratings_df = enriched_ratings_df.fillna({"title": "Unknown", "genre_id": -1, "genre_name": "Unknown", "release_year": -1, "genres_indexed": -1})

# ALS model configuration
als = ALS(
    maxIter=10,
    regParam=0.1,
    rank=5,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Split data into training and testing sets
(train, test) = ratings_df.randomSplit([0.8, 0.2], seed=42)

# Train the ALS model
model = als.fit(train)

# Model evaluation on test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"Root-Mean-Square Error (RMSE) on test data = {rmse}")

# Generate top 10 movie recommendations for each user
user_recommendations = model.recommendForAllUsers(10)

# Display recommendations with additional metadata
def show_recommendations_with_metadata(user_id, top_n=10):
    # Get recommendations for the specified user
    user_recs = user_recommendations.filter(col("userId") == user_id).select(explode("recommendations").alias("rec"))

    # Join with movie metadata for additional details
    user_recs = user_recs.selectExpr("rec.movieId as movieId", "rec.rating as predicted_rating")
    user_recs_with_metadata = user_recs.join(movies_df, on="movieId", how="left")

    # Fill NULL values in the metadata columns with "Unknown" or similar
    user_recs_with_metadata = user_recs_with_metadata.fillna({"title": "Unknown", "genre_id": -1, "genre_name": "Unknown", "release_year": -1, "genres_indexed": -1})
    
    # Display recommendations with metadata
    print(f"\nTop {top_n} Recommendations for User {user_id}:")
    user_recs_with_metadata.show(top_n, truncate=False)

# Example usage: Show recommendations with metadata for a sample user
show_recommendations_with_metadata(user_id=1, top_n=10)

# Stop Spark session
spark.stop()
