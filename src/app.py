import streamlit as st
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

# Import your recommendation functions here
from src.content_based_knn import get_similar_movies_knn  # Content-Based Filtering function
from src.collaborative_filtering_als import get_collaborative_recommendations  # Collaborative Filtering function
from src.hybrid_model import get_hybrid_recommendations  # Hybrid Model function
from src.collaborative_filtering_svd import get_knn_recommendations  # KNN-based function

# Define the main Streamlit app
def main():
    st.title("Movie Recommendation System")
    st.write("Welcome to the Movie Recommendation System!")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Choose a recommendation approach:", 
                               ("Content-Based Filtering", 
                                "Collaborative Filtering",
                                "KNN-based Recommendations", 
                                "Hybrid Recommendation"))

    # Content-Based Filtering
    if options == "Content-Based Filtering":
        st.header("Content-Based Movie Recommendations")
        movie_title = st.text_input("Enter a movie title:")
        top_n = st.slider("Number of recommendations:", 5, 20, 10)

        if st.button("Get Recommendations"):
            if movie_title:
                recommendations = get_similar_movies_knn(movie_title, top_n)
                st.write(f"Top {top_n} movies similar to '{movie_title}':")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning("Please enter a movie title.")

    # Collaborative Filtering
    elif options == "Collaborative Filtering":
        st.header("Collaborative Filtering Recommendations")
        user_id = st.text_input("Enter your user ID:")
        top_n = st.slider("Number of recommendations:", 5, 20, 10)

        if st.button("Get Recommendations"):
            if user_id:
                recommendations = get_collaborative_recommendations(user_id, top_n)
                st.write(f"Top {top_n} recommendations for user '{user_id}':")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning("Please enter your user ID.")

    # KNN-based Recommendations
    elif options == "KNN-based Recommendations":
        st.header("KNN-based Movie Recommendations")
        movie_title = st.text_input("Enter a movie title:")
        top_n = st.slider("Number of recommendations:", 5, 20, 10)

        if st.button("Get Recommendations"):
            if movie_title:
                recommendations = get_knn_recommendations(movie_title, top_n)
                st.write(f"Top {top_n} movies similar to '{movie_title}':")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning("Please enter a movie title.")

    # Hybrid Recommendation
    elif options == "Hybrid Recommendation":
        st.header("Hybrid Recommendations")
        movie_title = st.text_input("Enter a movie title:")
        user_id = st.text_input("Enter your user ID (optional):")
        top_n = st.slider("Number of recommendations:", 5, 20, 10)

        if st.button("Get Hybrid Recommendations"):
            if movie_title:
                recommendations = get_hybrid_recommendations(movie_title, user_id, top_n)
                st.write(f"Top {top_n} hybrid recommendations for movie '{movie_title}':")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning("Please enter a movie title.")

if __name__ == "__main__":
    main()
