from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, when, year, regexp_replace, length, log1p
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, BooleanType, ArrayType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CleanMoviesMetadata") \
    .config("spark.sql.warehouse.dir", "file:///usr/local/hive/warehouse") \
    .getOrCreate()

# Load movies_metadata.csv into a DataFrame
input_path = "hdfs:///user/data/movies_metadata.csv"
movies_df = spark.read.csv(input_path, header=True, inferSchema=True)

# Display initial schema and sample data
print("Initial Data Schema:")
movies_df.printSchema()
movies_df.show(5, truncate=False)

# Step 1: Convert data types
movies_df = movies_df \
    .withColumn("budget", regexp_replace("budget", "[^0-9]", "").cast(FloatType())) \
    .withColumn("popularity", regexp_replace("popularity", "[^0-9.]", "").cast(FloatType())) \
    .withColumn("id", regexp_replace("id", "[^0-9]", "").cast(IntegerType())) \
    .withColumn("imdb_id", col("imdb_id").cast(StringType())) \
    .withColumn("vote_count", col("vote_count").cast(IntegerType())) \
    .withColumn("vote_average", col("vote_average").cast(FloatType()))

# Step 2: Handle null values
movies_df = movies_df \
    .filter(col("id").isNotNull()) \
    .fillna({"budget": 0.0, "revenue": 0.0, "runtime": 0.0, "vote_count": 0}) \
    .fillna({"overview": "", "tagline": "", "title": ""}) \
    .filter(col("release_date").isNotNull()) \
    .fillna("Unknown", subset=["status", "original_language"])

# Step 3: Clean `adult` and `video` columns (should contain only "True" or "False")
movies_df = movies_df \
    .withColumn("adult", when(col("adult") == "True", True).otherwise(False).cast(BooleanType())) \
    .withColumn("video", when(col("video") == "True", True).otherwise(False).cast(BooleanType()))

# Step 4: Extract year from `release_date`
movies_df = movies_df.withColumn("release_year", year(col("release_date")))

# Step 5: Define schemas for JSON columns
json_schema_genres = ArrayType(StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True)
]))

json_schema_companies = ArrayType(StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True)
]))

json_schema_countries = ArrayType(StructType([
    StructField("iso_3166_1", StringType(), True),
    StructField("name", StringType(), True)
]))

json_schema_languages = ArrayType(StructType([
    StructField("iso_639_1", StringType(), True),
    StructField("name", StringType(), True)
]))

# Step 6: Parse and explode JSON columns
# Genres
movies_df = movies_df \
    .withColumn("genres", from_json(col("genres"), json_schema_genres)) \
    .withColumn("genres", explode("genres").alias("genre")) \
    .withColumn("genre_id", col("genres.id")) \
    .withColumn("genre_name", col("genres.name")) \
    .drop("genres")

# Production Companies
movies_df = movies_df \
    .withColumn("production_companies", from_json(col("production_companies"), json_schema_companies)) \
    .withColumn("production_companies", explode("production_companies").alias("company")) \
    .withColumn("company_id", col("production_companies.id")) \
    .withColumn("company_name", col("production_companies.name")) \
    .drop("production_companies")

# Production Countries
movies_df = movies_df \
    .withColumn("production_countries", from_json(col("production_countries"), json_schema_countries)) \
    .withColumn("production_countries", explode("production_countries").alias("country")) \
    .withColumn("country_iso", col("production_countries.iso_3166_1")) \
    .withColumn("country_name", col("production_countries.name")) \
    .drop("production_countries")

# Spoken Languages
movies_df = movies_df \
    .withColumn("spoken_languages", from_json(col("spoken_languages"), json_schema_languages)) \
    .withColumn("spoken_languages", explode("spoken_languages").alias("language")) \
    .withColumn("language_iso", col("spoken_languages.iso_639_1")) \
    .withColumn("language_name", col("spoken_languages.name")) \
    .drop("spoken_languages")

# Step 7: Remove duplicates based on 'id'
movies_df = movies_df.dropDuplicates(["id"])

# Additional Feature Engineering Steps

# Multi-hot encoding for genres
unique_genres = [row['genre_name'] for row in movies_df.select("genre_name").distinct().collect()]
for genre in unique_genres:
    movies_df = movies_df.withColumn(genre, when(col("genre_name") == genre, 1).otherwise(0))

# Log transformations for budget and revenue
movies_df = movies_df \
    .withColumn("log_budget", log1p("budget")) \
    .withColumn("log_revenue", log1p("revenue"))

# Binning popularity
popularity_quantiles = movies_df.approxQuantile("popularity", [0.33, 0.66], 0.05)
low_popularity, high_popularity = popularity_quantiles

movies_df = movies_df.withColumn(
    "popularity_bin",
    when(col("popularity") <= low_popularity, "low")
    .when((col("popularity") > low_popularity) & (col("popularity") <= high_popularity), "medium")
    .otherwise("high")
)

# Text length features
movies_df = movies_df \
    .withColumn("overview_length", length("overview")) \
    .withColumn("tagline_length", length("tagline"))

# Release decade
movies_df = movies_df.withColumn("release_decade", (col("release_year") / 10).cast(IntegerType()) * 10)

# Weighted rating
avg_vote_count = movies_df.agg({"vote_count": "avg"}).collect()[0][0]
movies_df = movies_df.withColumn(
    "weighted_rating",
    (col("vote_count") / (col("vote_count") + avg_vote_count) * col("vote_average")) +
    (avg_vote_count / (avg_vote_count + col("vote_count")) * 7.0)  # Assuming 7.0 as average rating
)

# Show cleaned and engineered data for verification
print("Cleaned and Engineered Movies Metadata Data:")
movies_df.show(5, truncate=False)

# Step 8: Write the cleaned and engineered data back to HDFS
output_path_movies = "hdfs:///user/data/cleaned_and_engineered_movies_metadata.csv"
movies_df.write.csv(output_path_movies, header=True, mode="overwrite")

# Stop Spark session
spark.stop()
