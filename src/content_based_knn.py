from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat_ws, lit
from pyspark.ml.feature import StringIndexer, Tokenizer, HashingTF, IDF, VectorAssembler
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("ContentBasedKNN").getOrCreate()

# Load movie metadata
movies_metadata_path = "hdfs:///user/data/cleaned_movies_metadata.csv"
movies_df = spark.read.csv(movies_metadata_path, header=True, inferSchema=True)

# Load additional metadata (e.g., cast, crew, keywords)
cast_df = spark.read.csv("hdfs:///user/data/cleaned_cast.csv", header=True, inferSchema=True)
crew_df = spark.read.csv("hdfs:///user/data/cleaned_crew.csv", header=True, inferSchema=True)
keywords_df = spark.read.csv("hdfs:///user/data/cleaned_keywords.csv", header=True, inferSchema=True)

# Rename 'id' column in movies_df to match the join key 'movie_id' in other DataFrames
movies_df = movies_df.withColumnRenamed("id", "movie_id")

# Print schema of each DataFrame to confirm column names
print("Movies DataFrame Schema:")
movies_df.printSchema()
print("Cast DataFrame Schema:")
cast_df.printSchema()
print("Crew DataFrame Schema:")
crew_df.printSchema()
print("Keywords DataFrame Schema:")
keywords_df.printSchema()

# Adjust column names based on actual DataFrame schema
# For example, if the cast column is named "lead_actor" in cast_df, use that instead
movies_df = (
    movies_df
    .join(cast_df.select("movie_id", "lead_actor"), on="movie_id", how="left")  # replace "cast" with the actual column name if needed
    .join(crew_df.select("movie_id", "most_common_department"), on="movie_id", how="left")  # replace "director" with actual column name if needed
    .join(keywords_df.select("movie_id", "keyword_name"), on="movie_id", how="left")  # replace "keywords" with actual column name if needed
)

# Fill nulls with empty strings in text columns to prevent errors
movies_df = movies_df.fillna({"overview": "", "keyword_name": "", "lead_actor": "", "most_common_department": ""})

# Combine multiple text columns into a single column for TF-IDF processing
movies_df = movies_df.withColumn("combined_text", concat_ws(" ", "overview", "keyword_name", "lead_actor", "most_common_department"))

# Tokenize and apply TF-IDF to 'combined_text'
tokenizer = Tokenizer(inputCol="combined_text", outputCol="text_tokens")
movies_df = tokenizer.transform(movies_df)

hashing_tf = HashingTF(inputCol="text_tokens", outputCol="text_tf", numFeatures=1000)
movies_df = hashing_tf.transform(movies_df)

idf = IDF(inputCol="text_tf", outputCol="text_tfidf")
idf_model = idf.fit(movies_df)
movies_df = idf_model.transform(movies_df)

# Encode categorical data: 'genre_name'
indexer = StringIndexer(inputCol="genre_name", outputCol="genre_indexed")
movies_df = indexer.fit(movies_df).transform(movies_df)

# Assemble all features into a single vector
assembler = VectorAssembler(
    inputCols=["text_tfidf", "genre_indexed"],  # You can add more relevant columns here
    outputCol="features"
)
movies_df = assembler.transform(movies_df)

# Extract feature vectors for each movie and convert to numpy array for KNN
feature_data = movies_df.select("movie_id", "title", "features").collect()
movie_ids = [row["movie_id"] for row in feature_data]
movie_titles = [row["title"] for row in feature_data]
feature_vectors = np.array([row["features"].toArray() for row in feature_data])

# Fit KNN model using cosine similarity
knn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='cosine')
knn.fit(feature_vectors)

# Define function to find similar movies using KNN
def get_similar_movies_knn(movie_title, top_n=10):
    # Find the index of the target movie
    try:
        movie_index = movie_titles.index(movie_title)
    except ValueError:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return
    
    # Get feature vector of the target movie
    target_vector = feature_vectors[movie_index].reshape(1, -1)
    
    # Find k-nearest neighbors
    distances, indices = knn.kneighbors(target_vector, n_neighbors=top_n + 1)  # +1 to include the movie itself
    
    # Filter out the movie itself from the results
    similar_movies = [(dist, idx) for dist, idx in zip(distances.flatten(), indices.flatten()) if idx != movie_index]
    
    # Display similar movies
    print(f"\nTop {top_n} movies similar to '{movie_title}':")
    for i, (dist, idx) in enumerate(similar_movies[:top_n]):
        print(f"{i+1}. {movie_titles[idx]} (Similarity: {1 - dist:.2f})")

# Example usage
get_similar_movies_knn("The Matrix", top_n=10)

# Stop Spark session
spark.stop()
