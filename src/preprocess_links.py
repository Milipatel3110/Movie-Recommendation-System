from pyspark.sql import SparkSession
from pyspark.sql.functions import col, format_string, length, when
from pyspark.sql.types import IntegerType, StringType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("CombineLinks") \
    .config("spark.sql.warehouse.dir", "file:///usr/local/hive/warehouse") \
    .getOrCreate()

# Load the CSV files into DataFrames
links_small_path = "hdfs:///user/data/links_small.csv"
links_path = "hdfs:///user/data/links.csv"

links_small_df = spark.read.csv(links_small_path, header=True, inferSchema=True)
links_df = spark.read.csv(links_path, header=True, inferSchema=True)

# Display initial data for verification
print("Links Small Data:")
links_small_df.show(5)
print("Links Data:")
links_df.show(5)

# Combine the two DataFrames using union
combined_links_df = links_small_df.union(links_df)

# Drop duplicates based on 'movieId'
combined_links_df = combined_links_df.dropDuplicates(["movieId"])

# Filter out rows with invalid or missing 'movieId'
combined_links_df = combined_links_df.filter(col("movieId").isNotNull())

# Fill missing 'tmdbId' with -1 and cast to IntegerType
combined_links_df = combined_links_df.fillna({"tmdbId": -1}).withColumn("tmdbId", col("tmdbId").cast(IntegerType()))

# Filter out rows where 'imdbId' or 'tmdbId' are less than or equal to zero (invalid IDs)
combined_links_df = combined_links_df.filter((col("imdbId") > 0) & (col("tmdbId") >= 0))

# Ensure 'imdbId' is also an integer for consistency
combined_links_df = combined_links_df.withColumn("imdbId", col("imdbId").cast(IntegerType()))

# Standardize 'imdbId' to be a 7-digit string
combined_links_df = combined_links_df.withColumn("imdbId", format_string("%07d", col("imdbId")))

# Feature Engineering Steps

# 1. Length of IMDb and TMDb IDs for analysis
combined_links_df = combined_links_df \
    .withColumn("imdbId_length", length("imdbId")) \
    .withColumn("tmdbId_length", length(col("tmdbId").cast(StringType())))

# 2. Binary encoding of presence for IMDb and TMDb IDs
combined_links_df = combined_links_df \
    .withColumn("has_imdbId", when(col("imdbId").isNotNull(), 1).otherwise(0)) \
    .withColumn("has_tmdbId", when(col("tmdbId") > 0, 1).otherwise(0))

# 3. Create a combined identifier feature (optional)
combined_links_df = combined_links_df.withColumn("combined_id", col("imdbId") + "_" + col("tmdbId").cast(StringType()))

# Show cleaned and engineered data for verification
print("Cleaned and Engineered Links Data:")
combined_links_df.show(5, truncate=False)

# Write the combined and cleaned data back to HDFS
output_path_combined_links = "hdfs:///user/data/combined_links.csv"
combined_links_df.write.csv(output_path_combined_links, header=True, mode="overwrite")

# Stop the Spark session
spark.stop()
