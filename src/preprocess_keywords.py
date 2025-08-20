from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, trim, regexp_replace, initcap

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PreprocessKeywords") \
    .config("spark.sql.warehouse.dir", "file:///usr/local/hive/warehouse") \
    .getOrCreate()

# HDFS file paths
input_path = "hdfs:///user/data/keywords.csv"
output_path_keywords = "hdfs:///user/data/cleaned_keywords.csv"

# Load keywords.csv into a DataFrame
keywords_df = spark.read.csv(input_path, header=True, inferSchema=True)
print("Initial Data:")
keywords_df.show(5, truncate=False)

# Split the 'keywords' column by commas to create an array of keywords
keywords_df = keywords_df.withColumn("keywords", split(col("keywords"), ","))

# Filter out rows where keywords is null or empty
keywords_df = keywords_df.filter(keywords_df.keywords.isNotNull())
print("After Splitting and Removing Null Keywords:")
keywords_df.show(5, truncate=False)

# Explode the array to get individual rows for each keyword
keywords_exploded_df = keywords_df.select(col("id").alias("movie_id"), explode(col("keywords")).alias("keyword_name"))
print("After Exploding Keywords:")
keywords_exploded_df.show(5, truncate=False)

# Clean up each keyword by removing leading/trailing quotes and whitespaces, and capitalize
keywords_cleaned_df = keywords_exploded_df \
    .withColumn("keyword_name", trim(regexp_replace(col("keyword_name"), "^['\"]|['\"]$", ""))) \
    .withColumn("keyword_name", initcap(col("keyword_name")))

# Remove duplicates based on unique identifiers for keywords
keywords_cleaned_df = keywords_cleaned_df.dropDuplicates(["movie_id", "keyword_name"])

# Drop rows with null or empty values in important columns
keywords_cleaned_df = keywords_cleaned_df.filter(col("keyword_name") != "")

# Show cleaned data for verification
print("Cleaned Keywords Data:")
keywords_cleaned_df.show(5, truncate=False)

# Write the cleaned data back to HDFS
keywords_cleaned_df.write.csv(output_path_keywords, header=True, mode="overwrite")

# Stop Spark session
spark.stop()
