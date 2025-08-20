Movie Recommendation System

A scalable and intelligent Movie Recommendation System built using Hadoop, Apache Spark, and AWS EC2, designed to handle large-scale datasets and provide personalized movie suggestions through hybrid recommendation models.

Motivation

With the exponential growth of movies and streaming content, finding the right movie quickly is a challenge. Our system addresses this by leveraging big data technologies and machine learning models to provide accurate, real-time, and user-friendly recommendations.

Objectives

Build a scalable recommendation system using Hadoop & Spark.

Provide content-based, collaborative, and hybrid recommendations.

Enable real-time data preprocessing and feature engineering.

Deploy a frontend interface with Streamlit for easy interaction.

Host the application on AWS for accessibility and performance.

Features

Multiple Recommendation Models:

Content-Based Filtering (genres, keywords, cast, director)

Collaborative Filtering (ALS & user-item interactions)

KNN-based Similarity Model

Hybrid Model (best of both worlds)

Exploratory Data Analysis (EDA): Genre popularity, rating distribution, insights via visualizations.

Scalable Data Handling: Distributed storage using HDFS.

Frontend UI: Movie discovery, trending lists, filters, personalized suggestions via Streamlit.

Deployment: Hosted on AWS EC2, ensuring high availability.

Tech Stack

Backend: Apache Spark (PySpark), Hadoop (HDFS)

Frontend: Streamlit

Cloud & Deployment: AWS EC2

Programming Language: Python

Dataset Source: Kaggle (Movies Metadata, Ratings, Credits, Keywords, Links)

Datasets Used

Movies Metadata (movies_metadata.csv) – Titles, genres, budget, revenue, runtime.

Ratings (ratings.csv, ratings_small.csv) – User ratings & timestamps.

Credits (credits.csv) – Cast & crew information.

Keywords (keywords.csv) – Movie keywords.

Links (links.csv, links_small.csv) – Mapping to IMDb and TMDb.

Workflow

Data Ingestion & Storage – Load datasets into HDFS.

Data Preprocessing – Type conversions, handling null values, feature engineering (genre encoding, log transformations, keyword parsing).

Model Training –

Content-Based Filtering

Collaborative Filtering (ALS)

KNN Similarity

Hybrid Recommendation

Evaluation – Metrics like RMSE, Precision, Recall, F1-score.

Frontend Deployment – Streamlit app for interactive recommendations.

Hosting – AWS EC2 instance with Git clone setup.

Results & Evaluation

RMSE: Evaluated model accuracy.

Precision & Recall: Measured relevance of recommendations.

Visualizations: Genre trends, rating patterns, user interaction insights.

Challenges Faced

Hadoop & Spark environment setup on Mac M1 (IP conflicts, disk space issues).

Managing JSON parsing and schema consistency across datasets.

Cold-start problems in collaborative filtering.

Memory & performance bottlenecks with high-dimensional features.

Synchronizing PySpark backend with Streamlit frontend.

Team Contributions

Khushi Desai: Project idea, objectives, collaborative/content-based models, implementation slides.

Miliben Patel: Features, preprocessing, feature engineering, hybrid model, frontend (Streamlit).

Azmaan Hemraj: Dataset collection, hosting on AWS, workflow diagrams, evaluation.

Imaduddin Ahmed: Data preprocessing, model evaluation (metrics & visualizations), documentation.

Installation & Usage
# Clone repository
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system

# Install dependencies
pip install -r requirements.txt

# Run Streamlit frontend
streamlit run app.py

Deployment on AWS

Launch an EC2 instance (Ubuntu).

Install dependencies (Python, Spark, Hadoop, Streamlit).

Clone the repository.

Run the app and configure security groups to allow HTTP access.

References

Apache Spark MLlib Documentation

Evaluation Metrics in ML – GeeksforGeeks

Amazon SageMaker Movie Recommender Blog

Collaborative Filtering Overview – Medium

This project combines the power of Big Data and AI to make movie discovery smarter, scalable, and user-friendly.
