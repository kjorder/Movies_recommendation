import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="TMDB Film Tavsiya Tizimi")

@st.cache_data
def load_data():
    df = pd.read_csv("tmdb-movies.csv")
    df['overview'] = df['overview'].fillna('')
    return df

@st.cache_resource
def build_model(texts):
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(texts)
    sim = cosine_similarity(tfidf_matrix)
    return sim

movies = load_data()
similarity = build_model(movies['overview'])

st.title("🎬 TMDB Film Tavsiya Tizimi")
movie_list = movies['title'].dropna().unique()
selected_movie = st.selectbox("Film nomini tanlang:", movie_list)

def recommend(movie_title, top_n=5):
    idx = movies[movies['title'] == movie_title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    rec_titles = movies.iloc[[i[0] for i in scores]]['title'].tolist()
    return rec_titles

if st.button("Tavsiya ber"):
    recs = recommend(selected_movie)
    st.subheader("🎥 Sizga yoqishi mumkin bo‘lgan filmlar:")
    for r in recs:
        st.write("•", r)
