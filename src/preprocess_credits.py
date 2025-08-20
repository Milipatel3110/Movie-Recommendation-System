from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, explode, initcap, count, first, collect_list, size, when
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType

# Initialize Spark session
spark = SparkSession.builder \
    .appName("PreprocessCredits") \
    .config("spark.sql.warehouse.dir", "file:///usr/local/hive/warehouse") \
    .getOrCreate()

# Define the schema for cast and crew JSON structure
cast_schema = ArrayType(StructType([
    StructField("cast_id", IntegerType(), True),
    StructField("character", StringType(), True),
    StructField("credit_id", StringType(), True),
    StructField("gender", IntegerType(), True),
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True),
    StructField("order", IntegerType(), True)
]))

crew_schema = ArrayType(StructType([
    StructField("credit_id", StringType(), True),
    StructField("department", StringType(), True),
    StructField("gender", IntegerType(), True),
    StructField("id", IntegerType(), True),
    StructField("job", StringType(), True),
    StructField("name", StringType(), True)
]))

# HDFS file paths
input_path = "hdfs:///user/data/credits.csv"
output_path_cast = "hdfs:///user/data/cleaned_cast.csv"
output_path_crew = "hdfs:///user/data/cleaned_crew.csv"

# Load credits.csv into a DataFrame
credits_df = spark.read.csv(input_path, header=True, inferSchema=True)

# Parse JSON columns
credits_df = credits_df \
    .withColumn("cast", from_json(col("cast"), cast_schema)) \
    .withColumn("crew", from_json(col("crew"), crew_schema))

# Explode the arrays to get individual rows for each cast and crew member
cast_df = credits_df.select(col("id").alias("movie_id"), explode(col("cast")).alias("cast_member"))
crew_df = credits_df.select(col("id").alias("movie_id"), explode(col("crew")).alias("crew_member"))

# Select relevant fields from cast and crew, with cleaning and transformation steps
cast_df = cast_df.select(
    "movie_id",
    col("cast_member.name").alias("cast_name"),
    col("cast_member.character"),
    col("cast_member.gender").alias("cast_gender"),
    col("cast_member.order").alias("cast_order")
)

crew_df = crew_df.select(
    "movie_id",
    col("crew_member.name").alias("crew_name"),
    col("crew_member.department"),
    col("crew_member.job"),
    col("crew_member.gender").alias("crew_gender")
)

# Data Cleaning Steps

# 1. Remove duplicates based on unique identifiers for cast and crew
cast_df = cast_df.dropDuplicates(["movie_id", "cast_name", "character"])
crew_df = crew_df.dropDuplicates(["movie_id", "crew_name", "job"])

# 2. Drop rows with null values in essential columns
cast_df = cast_df.dropna(subset=["movie_id", "cast_name"])
crew_df = crew_df.dropna(subset=["movie_id", "crew_name"])

# 3. Filter for valid gender values (only 0 or 1, or None if not specified)
cast_df = cast_df.filter((col("cast_gender").isNull()) | (col("cast_gender") >= 0))
crew_df = crew_df.filter((col("crew_gender").isNull()) | (col("crew_gender") >= 0))

# 4. Capitalize names for consistency
cast_df = cast_df.withColumn("cast_name", initcap(col("cast_name")))
crew_df = crew_df.withColumn("crew_name", initcap(col("crew_name")))

# Feature Engineering for Cast Data

# 1. Count the number of cast members per movie
cast_count_df = cast_df.groupBy("movie_id").agg(count("cast_name").alias("cast_count"))

# 2. Identify the lead actor (lowest order) per movie
lead_actor_df = cast_df.groupBy("movie_id").agg(first("cast_name", ignorenulls=True).alias("lead_actor"))

# 3. Gender distribution in cast (count males and females separately)
gender_distribution_cast_df = cast_df.groupBy("movie_id").agg(
    count(when(col("cast_gender") == 1, True)).alias("female_cast_count"),
    count(when(col("cast_gender") == 2, True)).alias("male_cast_count")
)

# Merge feature-engineered cast data
cast_features_df = cast_count_df.join(lead_actor_df, "movie_id", "left") \
    .join(gender_distribution_cast_df, "movie_id", "left")

# Feature Engineering for Crew Data

# 1. Count the number of crew members per movie
crew_count_df = crew_df.groupBy("movie_id").agg(count("crew_name").alias("crew_count"))

# 2. Most common department in crew per movie
common_department_df = crew_df.groupBy("movie_id").agg(
    first(col("department")).alias("most_common_department")
)

# 3. Gender distribution in crew (count males and females separately)
gender_distribution_crew_df = crew_df.groupBy("movie_id").agg(
    count(when(col("crew_gender") == 1, True)).alias("female_crew_count"),
    count(when(col("crew_gender") == 2, True)).alias("male_crew_count")
)

# Merge feature-engineered crew data
crew_features_df = crew_count_df.join(common_department_df, "movie_id", "left") \
    .join(gender_distribution_crew_df, "movie_id", "left")

# Show cleaned and feature-engineered data for verification
print("Feature Engineered Cast Data:")
cast_features_df.show(5, truncate=False)

print("Feature Engineered Crew Data:")
crew_features_df.show(5, truncate=False)

# Save the feature-engineered data back to HDFS
cast_features_df.write.csv(output_path_cast, header=True, mode="overwrite")
crew_features_df.write.csv(output_path_crew, header=True, mode="overwrite")

# Stop Spark session
spark.stop()
