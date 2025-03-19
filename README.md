# Music Emotion Classification for TikTok Videos

This project automatically recommends music for TikTok videos based on emotional analysis of both the video content and music tracks.

## Features

- Fetches music metadata from Spotify playlists
- Analyzes video content using Google Cloud Video Intelligence API
- Classifies music tracks into emotional categories
- Recommends music based on video content and emotional matching

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Spotify API credentials:
- Create a Spotify Developer account
- Create a new application
- Add `http://localhost:8888/callback` to Redirect URIs
- Copy Client ID and Client Secret to `recommend_spotify_playlist_music_for_tiktok_edits.py`

4. Set up Google Cloud credentials:
- Create a Google Cloud project
- Enable Video Intelligence API
- Create a service account and download credentials
- Set the environment variable: `GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json`

## Usage

Run the full pipeline:
```bash
python main.py --full-pipeline
```

Or run individual components:
```bash
# Fetch Spotify data
python main.py --fetch-spotify --playlist-id YOUR_PLAYLIST_ID

# Analyze video content
python main.py --analyze-video --bucket-name YOUR_BUCKET_NAME

# Train emotion classifier
python main.py --train-model

# Get music recommendations
python main.py --recommend
```

## Project Structure

- `main.py`: Main entry point and pipeline orchestration
- `AutoLabel.py`: Music emotion classification model
- `recommend_spotify_playlist_music_for_tiktok_edits.py`: Spotify data collection
- `GoogleVideoIntelligenceAPI.py`: Video content analysis

## Output

The system generates:
1. `spotify_metadata.xlsx`: Music track metadata from Spotify
2. `GoogleVideoIntelligenceLabelAnalyzer_results.xlsx`: Video content analysis
3. `emotion_classifier_model.h5`: Trained emotion classification model
4. Music recommendations based on video content

## Emotion Categories

The system classifies music into five emotional categories:
- Happy
- Sad
- Energetic
- Calm
- Aggressive

## License

MIT License 