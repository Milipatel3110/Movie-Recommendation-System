from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, avg, count
from pyspark.sql.types import IntegerType, FloatType

# Initialize Spark session
spark = SparkSession.builder.appName("CleanAndFeatureEngineerRatingsData").getOrCreate()

# Load the ratings_small and ratings datasets
ratings_small_df = spark.read.csv("hdfs:///user/data/ratings_small.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("hdfs:///user/data/ratings.csv", header=True, inferSchema=True)

# Check and unify schemas if necessary (although they are expected to match in this case)
ratings_small_df = ratings_small_df.select("userId", "movieId", "rating", "timestamp")
ratings_df = ratings_df.select("userId", "movieId", "rating", "timestamp")

# Combine the two datasets
combined_ratings_df = ratings_small_df.union(ratings_df)

# Data Cleaning Steps

# 1. Remove duplicate rows if any
combined_ratings_df = combined_ratings_df.dropDuplicates()

# 2. Remove rows with any null values (though these datasets should have no nulls based on info)
combined_ratings_df = combined_ratings_df.dropna()

# 3. Convert timestamp to a readable date format and add a new column 'rating_date'
combined_ratings_df = combined_ratings_df.withColumn("rating_date", from_unixtime(col("timestamp")).cast("date"))

# 4. Drop the timestamp column if it is no longer needed
combined_ratings_df = combined_ratings_df.drop("timestamp")

# 5. Ensure correct data types for each column
combined_ratings_df = combined_ratings_df.withColumn("userId", col("userId").cast(IntegerType()))
combined_ratings_df = combined_ratings_df.withColumn("movieId", col("movieId").cast(IntegerType()))
combined_ratings_df = combined_ratings_df.withColumn("rating", col("rating").cast(FloatType()))

# 6. Filter out any invalid ratings (e.g., if ratings should be between 0 and 5)
combined_ratings_df = combined_ratings_df.filter((col("rating") >= 0) & (col("rating") <= 5))

# Feature Engineering Steps

# 1. Calculate the average rating given by each user (user profile feature)
user_avg_rating_df = combined_ratings_df.groupBy("userId").agg(avg("rating").alias("user_avg_rating"))

# 2. Calculate the count of ratings given by each user (user engagement feature)
user_rating_count_df = combined_ratings_df.groupBy("userId").agg(count("rating").alias("user_rating_count"))

# 3. Calculate the average rating received by each movie (movie profile feature)
movie_avg_rating_df = combined_ratings_df.groupBy("movieId").agg(avg("rating").alias("movie_avg_rating"))

# 4. Calculate the count of ratings received by each movie (movie popularity feature)
movie_rating_count_df = combined_ratings_df.groupBy("movieId").agg(count("rating").alias("movie_rating_count"))

# Join the engineered features with the main ratings DataFrame

# Adding user-level features
combined_ratings_df = combined_ratings_df.join(user_avg_rating_df, on="userId", how="left")
combined_ratings_df = combined_ratings_df.join(user_rating_count_df, on="userId", how="left")

# Adding movie-level features
combined_ratings_df = combined_ratings_df.join(movie_avg_rating_df, on="movieId", how="left")
combined_ratings_df = combined_ratings_df.join(movie_rating_count_df, on="movieId", how="left")

# Show the schema and sample data to verify
combined_ratings_df.printSchema()
combined_ratings_df.show(5)

# Save the cleaned and feature-engineered data back to HDFS
combined_ratings_df.write.csv("hdfs:///user/data/cleaned_and_engineered_ratings.csv", header=True, mode="overwrite")

# Stop Spark session
spark.stop()
