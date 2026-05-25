import pandas as pd
import numpy as np

import joblib
import faiss
import streamlit as st

df = joblib.load("songs.pkl")
tfidf = joblib.load("tfidf.pkl")

vectors = tfidf.fit_transform(df["tags"]) 
dense_vector = vectors.toarray().astype("float32") # Because FAISS need array
faiss.normalize_L2(dense_vector)
dimention = dense_vector.shape[1]
index = faiss.IndexFlatIP(dimention)
index.add(dense_vector)


st.set_page_config(
    page_title= "Spotify Song Reccomendar"
)
st.image("images/banner_2.png.png", width= 1000)

st.header("Song Recommender System")
st.write("Find song similar to your Request")




st.sidebar.title("About")

st.sidebar.info(
    """
    A machine learning based music recommendation system 
    that suggests similar songs using content similarity.

    Built with TF-IDF vectorization, FAISS vector search, 
    and Streamlit for fast and scalable recommendations.

    The suggesions are based on the data set used to make 
    this project, so it may not have all the songs that are available on spotify
    """
)



def recommendar(song):
    song = song.lower().strip()

    match_song = df[df["track_name"].str.lower().str.strip() == song]
    if match_song.empty:
        return []


    song_index = int(match_song.index[0])

    query_vector = index.reconstruct(song_index).reshape(1,-1)

    distances , indices = index.search(query_vector,6)

    recommended_songlist = []

    for el in indices[0][:]:
        recommended_songlist.append({
        "song_name" : df.iloc[el]["track_name"],
        "song_track_id" : df.iloc[el]["track_id"],
        "artist_name" : df.iloc[el]["artists"],
        "spotify_link" : f"https://open.spotify.com/track/{df.iloc[el]['track_id']}"}
        )

    return recommended_songlist
songs_list = df["track_name"].str.lower().str.strip().unique()

selected_song = st.selectbox(
    "Srarch Song",
    songs_list,
    index=None,
    placeholder="Enter Song"
)


if st.button("Recommend"):

    recommendations = recommendar(selected_song)
    if len(recommendations) == 0:
        st.error("Song Not Found ")
    
    else:
        st.subheader("Recommended Songd -->")
        for i, el in enumerate(recommendations):
            st.write(f'{i+1}: {el["song_name"]},{el["spotify_link"]} ')
            st.write("---------------------------------")
    st.success("Done!")

