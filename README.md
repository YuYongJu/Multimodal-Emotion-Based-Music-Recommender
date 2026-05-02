# Music Emotion Classification System

End-to-end multimodal pipeline that takes a video, classifies its emotional
context via Google Cloud Video Intelligence, and recommends emotionally
matching music tracks from a Spotify playlist using real audio features
and a Keras emotion classifier.

## Pipeline

```
Spotify playlist  →  audio features  →  emotion classifier (Keras)
                                                ↓
GCS bucket video  →  Video Intelligence labels  →  emotion target
                                                ↓
                                       ranked recommendations
```

## Audio features — important note on Spotify API status

Spotify deprecated the public `audio-features` endpoint for new third-party
applications on 2024-11-27. This project handles that with a two-source
extractor (`audio_features.py`):

1. **Spotify Web API** — used first when the app credentials still have
   access (apps registered before the deprecation date typically do).
2. **Local DSP via librosa** — fallback that downloads the 30-second
   preview MP3 exposed by `track.preview_url` and computes Spotify-shaped
   features locally (RMS energy, beat-track tempo, chroma key/mode,
   spectral centroid as valence proxy, zero-crossing rate as speechiness,
   spectral flatness as instrumentalness, RMS variance as liveness).

If neither source is available for a given track, the track is dropped
rather than substituted with fabricated values. This is a deliberate
design choice — earlier versions of this code used `random.uniform()` to
fill in features, which made the model train on noise.

## Setup

```bash
git clone https://github.com/YuYongJu/Multimodal-Emotion-Based-Music-Recommender.git
cd Multimodal-Emotion-Based-Music-Recommender
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file (do not commit it):

```env
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
PLAYLIST_IDS=37i9dQZF1DXcBWIGoYBM5M
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
BUCKET_NAME=your-gcs-bucket-with-mp4s
```

The `GOOGLE_APPLICATION_CREDENTIALS` and `BUCKET_NAME` variables are only
needed when running `--analyze-video`. Spotify-only operations work
without them.

## Usage

```bash
python main.py --fetch-spotify                 # build metadata + features xlsx
python main.py --analyze-video                 # run Video Intelligence on bucket
python main.py --train-model                   # train classifier on real features
python main.py --recommend                     # rank tracks for the analyzed video
python main.py --full-pipeline                 # run all four in order
python main.py --fetch-spotify --playlist-id <ID>
```

## Files

- `main.py` — CLI orchestration, lazy-init for both API clients
- `audio_features.py` — Spotify API → librosa fallback feature extractor
- `recommend_spotify_playlist_music_for_tiktok_edits.py` — playlist fetch with feature attachment
- `AutoLabel.py` — `MusicEmotionClassifier` (Keras MLP, real features)
- `GoogleVideoIntelligenceAPI.py` — GCS + Video Intelligence wrapper, lazy clients

The `emotion_classifier_model.h5` and `emotion_scaler.pkl` artifacts are
not committed because earlier checked-in versions were trained on
fabricated features and had no predictive value. Run `--train-model`
after a fresh `--fetch-spotify` to produce honest weights.

## License

MIT — see [LICENSE](LICENSE).
