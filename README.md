# Spotify_Recommendation_System


## Description

The Spotify Recommendation System is a project designed to provide personalized music recommendations based on user playlists or specific songs. By leveraging Spotify's API and various machine learning techniques, this system aims to enhance the music discovery experience for users.

## Data extraction
The first step will be to obtain keys to use. We'll need a [Spotify for developers](https://developer.spotify.com/) account for this. This is equivalent to a Spotify account and does not necessitate Spotify Premium. Go to the dashboard and select "create an app" from there. We now have access to the public and private keys required to use the API.

Now that we have an app, we can get a client ID and a client secret for this app. Both of these will be required to authenticate with the Spotify web API for our application, and can be thought of as a kind of username and password for the application. It is best practice not to share either of these, but especially don’t share the client secret key. To prevent this, we can keep it in a separate file, which, if you’re using Git for version control, should be Gitignored.

Spotify credentials should be stored the in the a `Spotify.yaml` file with the first line as the **credential id** and the second line as the **secret key**:
```python
Client_id : ************************
client_secret : ************************
```
To access this credentials, please use the following code:
```python
stream= open("Spotify/Spotify.yaml")
spotify_details = yaml.safe_load(stream)
auth_manager = SpotifyClientCredentials(client_id=spotify_details['Client_id'],
                                        client_secret=spotify_details['client_secret'])
sp = spotipy.client.Spotify(auth_manager=auth_manager)
```

# STEPS:

## STEP 01: Clone the repository

```bash
git clone https://github.com/charan-613/Spotify_Recommendation_System.git
```

## STEP 02: create and activate conda environment

```bash
conda create -n spotify_recomm python=3.9 -y
conda activate spotify_recomm
```

## STEP 03: install requirements.txt
```bash
pip install -r requirements.txt
```

## Run the app.py
```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```

# Dataset
I've used the Spotify Million Playlist Dataset: https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge

I've used Content Based Recommendation System for the above project.
![image](https://user-images.githubusercontent.com/107134115/201203569-6bcd14fd-6704-4ad7-9577-44095bd65f74.png)

### Reference
- https://medium.com/analytics-vidhya/music-recommender-system-part-2-ff4c3f54cba3
