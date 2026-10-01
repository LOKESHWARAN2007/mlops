import ast
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def names(text, limit=None, job=None):
    out = []
    for d in ast.literal_eval(text):
        if job and d.get("job") != job:
            continue
        out.append(d["name"].replace(" ", ""))
        if limit and len(out) == limit:
            break
    return out

@st.cache_resource
def load():
    m = pd.read_csv("data/tmdb_5000_movies.csv")
    c = pd.read_csv("data/tmdb_5000_credits.csv")
    df = (m.merge(c, on="title")
           [["movie_id", "title", "overview", "genres", "keywords", "cast", "crew"]]
           .dropna().reset_index(drop=True))
    tags = df.apply(lambda r: " ".join(
        r.overview.split() + names(r.genres) + names(r.keywords)
        + names(r.cast, limit=3) + names(r.crew, job="Director")).lower(), axis=1)
    vectors = CountVectorizer(max_features=5000, stop_words="english").fit_transform(tags)
    return df["title"].tolist(), vectors

titles, vectors = load()

def recommend(title, n=5):
    i = titles.index(title)
    sims = cosine_similarity(vectors[i], vectors).ravel()   # one row, not a full matrix
    return [titles[j] for j in sims.argsort()[::-1][1:n + 1]]

st.title("Movie Recommender")
choice = st.selectbox("Pick a movie", titles)
if st.button("Recommend"):
    for t in recommend(choice):
        st.write("•", t)
