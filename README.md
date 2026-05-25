# Spotify Song Recommender Version 2

![Banner](banner_2.png.png)
Link ==> ""
This is the Second Version for song reccomendation System.
A scalable content-based music recommendation system built using TF-IDF vectorization and FAISS vector similarity search.

## Features

- Fast song similarity search
- Content-based recommendation system
- Spotify track links integration
- Scalable FAISS vector indexing
- Streamlit web interface

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- FAISS
- Pandas
- NumPy

## How It Works

The system combines song metadata such as:
- Track Name
- Artist Name
- Album Name
- Genre

These features are converted into TF-IDF vectors and indexed using FAISS for efficient similarity search across large datasets.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Project Highlights

- Improved scalability over traditional KNN-based systems
- Optimized for large music datasets
- Faster recommendation retrieval using vector search

## Author

Rishi Dubey