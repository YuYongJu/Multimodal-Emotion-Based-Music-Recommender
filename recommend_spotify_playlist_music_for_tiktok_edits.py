import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd
import sys

# Spotify API credentials
# You need to replace these with your own credentials from your Spotify Developer account
# Get them from https://developer.spotify.com/dashboard/
SPOTIFY_CLIENT_ID = '25bb4432c3f04e3d9744c141278c13d1'  # Replace with your client ID
SPOTIFY_CLIENT_SECRET = 'fe163149e2f44693b7ff8aa3a44f3670'  # Replace with your client secret
REDIRECT_URI = 'http://localhost:8888/callback'

# Create a Spotify client with proper authentication
try:
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope='playlist-read-private playlist-read-collaborative'
    ))
    print("Successfully connected to Spotify API!")
except Exception as e:
    print(f"Error connecting to Spotify API: {e}")
    print("Please check your credentials and internet connection.")
    sys.exit(1)

# Function to search for playlists
def search_playlists(query, limit=5):
    try:
        print(f"Searching for playlists with query: '{query}'")
        results = sp.search(q=query, type='playlist', limit=limit)
        
        if not results or 'playlists' not in results or 'items' not in results['playlists']:
            print(f"No valid playlist results found for query: '{query}'")
            return []
            
        playlists = results['playlists']['items']
        
        if not playlists:
            print(f"No playlists found for query: '{query}'")
            return []
            
        print(f"Found {len(playlists)} playlists:")
        valid_playlists = []
        
        for i, playlist in enumerate(playlists):
            try:
                # Only include playlists with required fields
                if 'id' in playlist and 'name' in playlist:
                    playlist_name = playlist['name']
                    playlist_id = playlist['id']
                    
                    # Get owner display name safely
                    owner_name = "Unknown user"
                    if 'owner' in playlist and playlist['owner'] and 'display_name' in playlist['owner']:
                        owner_name = playlist['owner']['display_name']
                    
                    # Get track count safely
                    track_count = 0
                    if 'tracks' in playlist and playlist['tracks'] and 'total' in playlist['tracks']:
                        track_count = playlist['tracks']['total']
                    
                    print(f"{i+1}. {playlist_name} (ID: {playlist_id}) by {owner_name} - {track_count} tracks")
                    valid_playlists.append(playlist)
            except Exception as e:
                print(f"Error processing playlist {i+1}: {e}")
                continue
                
        return valid_playlists
    except Exception as e:
        print(f"Error searching for playlists: {e}")
        return []

# Function to get Spotify metadata for a given track
def get_track_metadata(track_id):
    try:
        track = sp.track(track_id)
        metadata = {
            'track_id': track_id,
            'track_name': track['name'],
            'artist': track['artists'][0]['name'],
            'album_name': track['album']['name'],
            'release_date': track['album']['release_date'],
            'duration_ms': track['duration_ms'],
            'popularity': track['popularity'],
            'preview_url': track.get('preview_url', '')
        }
        return metadata
    except Exception as e:
        print(f"Error getting metadata for track {track_id}: {e}")
        return None

# Function to fetch Spotify metadata for songs in a playlist
def fetch_spotify_metadata(playlist_id):
    try:
        print(f"Fetching playlist with ID: {playlist_id}")
        
        # Get tracks from the playlist
        offset = 0
        tracks = []
        while True:
            try:
                results = sp.playlist_items(playlist_id, fields="items(track(id))", offset=offset)
                if not results['items']:
                    break
                tracks.extend(results['items'])
                offset += len(results['items'])
                print(f"Fetched {len(tracks)} tracks so far...")
            except Exception as e:
                print(f"Error fetching playlist items at offset {offset}: {e}")
                break

        if not tracks:
            print("No tracks found in playlist.")
            return pd.DataFrame()

        track_ids = [track['track']['id'] for track in tracks if track['track']]
        print(f"Found {len(track_ids)} valid track IDs")
        
        # Limit to 20 tracks for testing
        if len(track_ids) > 20:
            print("Limiting to 20 tracks for faster processing...")
            track_ids = track_ids[:20]

        # List to store the metadata
        metadata_list = []

        # Iterate over each track and fetch Spotify metadata
        for i, track_id in enumerate(track_ids):
            print(f"Processing track {i+1}/{len(track_ids)}: {track_id}")
            track_metadata = get_track_metadata(track_id)
            if track_metadata:
                metadata_list.append(track_metadata)

        # Create a dataframe from the metadata list
        df = pd.DataFrame(metadata_list)
        
        print(f"Successfully processed {len(metadata_list)} tracks")
        return df
    except Exception as e:
        print(f"Error in fetch_spotify_metadata: {e}")
        return pd.DataFrame()

# Main execution
try:
    print("Starting Spotify metadata extraction...")
    
    # Fallback playlist IDs from Spotify (Verified to work in most regions)
    # Today's Top Hits, Spotify Global Top 50, Global Viral 50
    fallback_playlist_ids = [
        ("37i9dQZF1DXcBWIGoYBM5M", "Today's Top Hits"),
        ("37i9dQZF1DXcBWIGoYBM5M", "Top 50 - Global"),
        ("37i9dQZF1DXa2EiKmMLhFD", "Release Radar"),
        ("37i9dQZEVXbNG2KDcFcKOF", "Spotify Viral 50")
    ]
    
    # Try the search approach first
    playlist_id = None
    playlist_name = None
    
    # Search for pop playlists
    playlists = search_playlists("pop")
    
    if not playlists:
        print("No playlists found with 'pop'. Trying 'hits'...")
        playlists = search_playlists("hits")
    
    if playlists:
        # Use the first playlist from search results
        selected_playlist = playlists[0]
        playlist_id = selected_playlist['id']
        playlist_name = selected_playlist['name']
        print(f"Using playlist from search: {playlist_name} (ID: {playlist_id})")
    else:
        # Try fallback playlists
        print("No playlists found via search. Trying fallback playlists...")
        
        for fallback_id, fallback_name in fallback_playlist_ids:
            try:
                # Test if the playlist exists
                test = sp.playlist(fallback_id, fields="id,name")
                playlist_id = fallback_id
                playlist_name = test.get('name', fallback_name)
                print(f"Using fallback playlist: {playlist_name} (ID: {playlist_id})")
                break
            except Exception as e:
                print(f"Fallback playlist {fallback_name} not accessible: {e}")
                continue
    
    if not playlist_id:
        print("Could not find any accessible playlists. Exiting.")
        sys.exit(1)
    
    # Fetch metadata from the selected playlist
    df = fetch_spotify_metadata(playlist_id)
    
    if df.empty:
        print("No data was fetched. Exiting.")
        sys.exit(1)
        
    # Export the dataframe to an Excel file
    safe_name = ''.join(c if c.isalnum() or c == ' ' else '_' for c in playlist_name)
    output_file = f'spotify_metadata_{safe_name.replace(" ", "_")}.xlsx'
    df.to_excel(output_file, index=False)
    print("Data exported to", output_file)
except Exception as e:
    print(f"Error in main execution: {e}")
    sys.exit(1)
