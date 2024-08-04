import streamlit as st
import pandas as pd
import numpy as np
import re
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the KMeans model
with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to load dataset
@st.cache_data
def load_data():
    path = r'C:\Users\chara\Downloads\sptfy\spotify_data_scaled.csv'  # Use raw string for Windows path
    df = pd.read_csv(path)
    return df

# Load dataset
df = load_data()

# Replace with your actual Spotify API credentials
client_id = "af589e660451438184e20e35de29702b"
client_secret = "a7126e3469514fcab5be79000be8cb6d"

# Setting up Spotipy
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret))

# Defining function for finding song on Spotify
def find_song(name, artists):
    song_data = defaultdict()  # Setting up song dictionary

    song_results = sp.search(q= 'track: {} artist: {}'.format(name, artists), limit=1)  # Searching for song data using song and artist name
    if song_results['tracks']['items'] == []:
        return None

    artist_results = sp.search(q= 'track: {} artist: {}'.format(name, artists), type='artist', limit=1)  # Searching for artist data using song and artist name

    results = song_results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]  # Getting song audio features

    song_data['name'] = [name]
    song_data['release_year'] = [pd.to_datetime(results['album']['release_date']).year]
    song_data['explicit'] = [int(results['explicit'])]
    genres = artist_results['artists']['items'][0]['genres']
    if genres != []:
        song_data['artist_genres_encoded'] = [label_encoder.transform([genres[0]])]
    else:
        song_data['artist_genres_encoded'] = [label_encoder.transform(['unknown'])]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

# Selecting training features (numerate columns)
number_cols = df.select_dtypes(np.number).columns

def get_song_data(song, spotify_data):
    try:  # Try to find song in imported data
        song_data = spotify_data[(spotify_data['name'] == song['name'])
                                & (spotify_data['artists'] == song['artists'])].iloc[0]
        return song_data

    except IndexError:  # If song not found, it will be retrieved from Spotify
        return find_song(song['name'], song['artists'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []

    for song in song_list:
        song_data = get_song_data(song, spotify_data)  # Getting song data from user input
        if song_data is None:
            print('Warning: {} does not exist in Spotify or in database'.format(song['name']))  # Returning Warning if song does not exist in Spotify
            continue

        # Ensure the columns are available before accessing
        missing_cols = [col for col in number_cols if col not in song_data.index]
        if missing_cols:
            print(f"Missing columns in song_data: {missing_cols}")
            continue

        song_vector = song_data[number_cols].values  # Getting audio features of the data
        song_vectors.append(song_vector)

    song_matrix = np.array(list(song_vectors))
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict()

    # Assign keys for dictionary
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    # Append data to the dictionary
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict

# Function to extract track ID from Spotify URL
def extract_track_id(url):
    return url.split("/")[-1].split("?")[0]

# Function to get song data from Spotify URL
def get_song_data_from_url(url):
    track_id = extract_track_id(url)
    track_data = sp.track(track_id)
    name = track_data['name']
    artists = ', '.join([artist['name'] for artist in track_data['artists']])
    return {'name': name, 'artists': artists}

# Modified recommend_songs function
# Modified recommend_songs function
def recommend_songs(song_list, spotify_data, n_songs=20):
    metadata_cols = ['name', 'artists', 'album_name', 'release_year', 'artist_genres']  # Features to be returned with recommendations

    # Applying pre-processing functions
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)

    # Scaling data
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))

    # Computing distances
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs*2][0])  # Get more indices to filter for artist

    # Getting recommended songs from the dataset
    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]

    # Filter to ensure the first 5 recommendations are by the same artist
    target_artist = song_list[0]['artists']
    same_artist_recommendations = rec_songs[rec_songs['artists'].str.contains(target_artist)].head(5)
    other_recommendations = rec_songs[~rec_songs['artists'].str.contains(target_artist)]
    rec_songs = pd.concat([same_artist_recommendations, other_recommendations]).head(n_songs)

    # Convert recommendations to a dictionary
    rec_songs_dict = rec_songs[metadata_cols].to_dict(orient='records')

    # Get additional recommendations from Spotify
    spotify_recommendations = []
    for song in song_list:
        track_id = sp.search(q=f'track:{song["name"]} artist:{song["artists"]}', type='track', limit=1)
        if track_id['tracks']['items']:
            track_id = track_id['tracks']['items'][0]['id']
            rec_tracks = sp.recommendations(seed_tracks=[track_id], limit=n_songs)['tracks']
            for track in rec_tracks:
                name = track['name']
                artists = ', '.join([artist['name'] for artist in track['artists']])
                album_name = track['album']['name']
                release_year = pd.to_datetime(track['album']['release_date']).year
                artist_genres = sp.artist(track['artists'][0]['id'])['genres']
                artist_genres = artist_genres[0] if artist_genres else "unknown"
                spotify_recommendations.append({
                    'name': name,
                    'artists': artists,
                    'album_name': album_name,
                    'release_year': release_year,
                    'artist_genres': artist_genres
                })

    # Combine dataset and Spotify recommendations
    combined_recommendations = rec_songs_dict + spotify_recommendations
    combined_recommendations = [dict(t) for t in {tuple(d.items()) for d in combined_recommendations}]
    combined_recommendations = sorted(combined_recommendations, key=lambda x: x['release_year'], reverse=True)[:n_songs]

    return combined_recommendations

# Streamlit App
st.title("Spotify Music Recommendation System")

# Option to use example URL or input URL
option = st.selectbox("Choose input method", ["Enter Spotify URL", "Use Example URL"])

if option == "Use Example URL":
    # Static example URL
    song_url = "https://open.spotify.com/track/0anBHV3eP4hiHsLQR5AZsm?si=f817441024654afc"  # Example URL
    song_data = get_song_data_from_url(song_url)
    recommendations = recommend_songs([song_data], df)
    st.write("Recommendations based on example URL:")
    st.write(pd.DataFrame(recommendations))

else:
    # Input field for Spotify URL
    spotify_url = st.text_input("Enter the Spotify track URL:")
    
    if st.button("Get Recommendations"):
        track_id = extract_track_id(spotify_url)
        if track_id:
            song_data = get_song_data_from_url(spotify_url)
            song_list = [{'name': song_data['name'], 'artists': song_data['artists'], 'track_id': track_id}]
            recommendations = recommend_songs(song_list, df)
            st.write("Recommendations based on input URL:")
            st.write(pd.DataFrame(recommendations))
        else:
            st.error("Invalid Spotify URL.")


