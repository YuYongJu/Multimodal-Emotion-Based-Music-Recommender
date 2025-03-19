# Music Emotion Classification System

An AI-powered system that analyzes video content and recommends emotionally matching music using Google Cloud Video Intelligence and Spotify APIs.

## Features

- **Video Content Analysis**: Uses Google Cloud Video Intelligence API to analyze video content and extract meaningful labels and categories
- **Music Emotion Classification**: Neural network model that classifies music tracks into emotional categories
- **Smart Recommendations**: Recommends music based on emotional matching between video content and music tracks
- **Spotify Integration**: Fetches music metadata from Spotify playlists
- **Environment Variable Support**: Secure credential management using .env files

## Prerequisites

- Python 3.8+
- Google Cloud Platform account with Video Intelligence API enabled
- Spotify Developer account with API credentials
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Music-Emotion-Classification.git
cd Music-Emotion-Classification
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your credentials:
     ```
     SPOTIFY_CLIENT_ID=your_spotify_client_id
     SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
     GOOGLE_APPLICATION_CREDENTIALS=path_to_your_credentials.json
     PLAYLIST_IDS=your_playlist_ids
     BUCKET_NAME=your_bucket_name
     ```

## Usage

The system can be run in different modes:

1. **Fetch Spotify Data**:
```bash
python main.py --fetch-spotify
```

2. **Analyze Video Content**:
```bash
python main.py --analyze-video
```

3. **Train Emotion Classifier**:
```bash
python main.py --train-model
```

4. **Get Music Recommendations**:
```bash
python main.py --recommend
```

5. **Run Full Pipeline**:
```bash
python main.py --full-pipeline
```

## Project Structure

- `main.py`: Main script orchestrating the entire system
- `AutoLabel.py`: Music emotion classification model
- `GoogleVideoIntelligenceAPI.py`: Video content analysis using Google Cloud
- `recommend_spotify_playlist_music_for_tiktok_edits.py`: Spotify playlist processing
- `.env`: Environment variables and credentials (not committed to git)
- `requirements.txt`: Python package dependencies

## Security Best Practices

- Credentials are stored in `.env` file (not committed to git)
- Google Cloud credentials are managed securely
- API keys are never exposed in the code
- Sensitive data is excluded from version control

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Cloud Video Intelligence API
- Spotify Web API
- TensorFlow for machine learning capabilities 