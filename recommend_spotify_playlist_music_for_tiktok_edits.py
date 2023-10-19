import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd

# Spotify API credentials
SPOTIFY_CLIENT_ID = '25bb4432c3f04e3d9744c141278c13d1'
SPOTIFY_CLIENT_SECRET = 'fe163149e2f44693b7ff8aa3a44f3670'

# Create a Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Function to get Spotify metadata for a given track
def get_track_metadata(track_id):
    track = spotify.track(track_id)
    audio_features = spotify.audio_features(track_id)
    audio_analysis = spotify.audio_analysis(track_id)
    metadata = {
        'track_name': track['name'],
        'artist': track['artists'][0]['name'],
        'album_name': track['album']['name'],
        'release_date': track['album']['release_date'],
        'duration_ms': track['duration_ms'],
        'popularity': track['popularity'],
        'audio_features': audio_features[0],
        'audio_analysis': audio_analysis,
        # Add more metadata fields as needed
    }
    return metadata

# Function to fetch Spotify metadata for songs in a playlist
def fetch_spotify_metadata(playlist_id):
    # Get tracks from the playlist
    offset = 0
    tracks = []
    while True:
        results = spotify.playlist_items(playlist_id, fields="items(track(id))", offset=offset)
        if not results['items']:
            break
        tracks.extend(results['items'])
        offset += len(results['items'])

    track_ids = [track['track']['id'] for track in tracks if track['track']]

    # List to store the metadata
    metadata_list = []

    # Iterate over each track and fetch Spotify metadata
    for track_id in track_ids:
        # Fetch Spotify metadata for the current track
        track_metadata = get_track_metadata(track_id)

        # Append the metadata to the list
        metadata_list.append(track_metadata)

    # Create a dataframe from the metadata list
    df = pd.DataFrame(metadata_list)

    return df


# Call the function to fetch Spotify metadata for songs in a playlist
playlist_id = '65LdqYCLcsV0lJoxpeQ6fW'
df = fetch_spotify_metadata(playlist_id)

# Export the dataframe to an Excel file
output_file = 'spotify_metadata.xlsx'
df.to_excel(output_file, index=False)

print("Data exported to", output_file)
