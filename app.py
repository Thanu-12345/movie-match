import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="Movie Match",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Match")
st.write("Discover movies similar to your favorite films!")

# Load saved files
@st.cache_resource
def load_data():
    with open("models/tfidf_matrix.pkl", "rb") as file:
        tfidf_matrix = pickle.load(file)

    movies = pd.read_csv("models/movies_with_ratings.csv")

    cosine_sim = cosine_similarity(
        tfidf_matrix,
        tfidf_matrix
    )

    return movies, cosine_sim


movies, cosine_sim = load_data()

# Movie selection
movie_titles = movies["title"].tolist()

selected_movie = st.selectbox(
    "Select a movie",
    movie_titles
)

num_recommendations = st.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

if st.button("🎯 Recommend Movies"):

    movie_index = movies[
        movies["title"] == selected_movie
    ].index[0]

    similarity_scores = list(
        enumerate(cosine_sim[movie_index])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for index, similarity in similarity_scores[1:]:
        movie = movies.iloc[index]

        recommendations.append({
            "Movie": movie["title"],
            "Genres": movie["genres"],
            "Rating": round(movie["average_rating"], 2),
            "Similarity": round(similarity, 3)
        })

        if len(recommendations) == num_recommendations:
            break

    recommendations_df = pd.DataFrame(recommendations)

    st.subheader("🍿 Recommended Movies")
    st.dataframe(
        recommendations_df,
        use_container_width=True,
        hide_index=True
    )