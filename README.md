# Music Emotion Classification System

End-to-end multimodal pipeline that takes a video, classifies its emotional
context via Google Cloud Video Intelligence, extracts real audio features
from a Spotify playlist, and recommends emotionally matching music tracks
through a trained Keras emotion classifier.

## Pipeline

```
Spotify playlist  →  audio features  →  emotion classifier (Keras MLP)
                                                ↓
GCS bucket video  →  Video Intelligence labels  →  emotion target
                                                ↓
                                       ranked recommendations
```

## Headline numbers

Reproducible from the committed benchmark dataset:

| Metric | Value |
| --- | ---: |
| Test accuracy (5-class emotion, held-out 2.1K) | **97.4%** |
| Per-class F1 (min across happy/sad/energetic/calm/aggressive) | 0.94 |
| Inference throughput (batch=1024, CPU) | **68,649 tracks/s** |
| Per-track latency (batch=1 → batch=1024) | 13.3 ms → **14.6 µs** (910× faster) |
| End-to-end pipeline (10K tracks, naive → optimized) | 2.32 s → **22 ms** (103× faster) |

Full reports: [`docs/EVALUATION.md`](docs/EVALUATION.md), [`results/evaluation.json`](results/evaluation.json), [`results/benchmark.json`](results/benchmark.json), [`results/confusion_matrix.png`](results/confusion_matrix.png).

## Reproduce in three commands

The repo ships with a synthetic 10,500-track / 1,021-segment benchmark
dataset and pre-trained model weights, so the headline numbers above
can be regenerated end-to-end without any API credentials:

```bash
pip install -r requirements.txt
python3 scripts/evaluate.py     # reproduces accuracy + confusion matrix
python3 scripts/benchmark.py    # reproduces latency + throughput numbers
```

## Audio features — Spotify API status

Spotify deprecated the public `audio-features` endpoint for new third-party
applications on 2024-11-27. This project handles that with a two-source
extractor (`audio_features.py`):

1. **Spotify Web API** — used first when the app credentials still have
   access (apps registered before the deprecation date typically do).
2. **Local DSP via librosa** — fallback that downloads the 30-second
   preview MP3 exposed by `track.preview_url` and computes Spotify-shaped
   features locally (RMS energy, beat-track tempo, chroma key/mode,
   spectral centroid as valence proxy, zero-crossing rate as speechiness,
   spectral flatness as instrumentalness).

If neither source is available for a track, the track is dropped rather
than substituted with fabricated values. Earlier versions of this code
used `random.uniform()` for features at runtime — see [`docs/ITERATION_LOG.md`](docs/ITERATION_LOG.md)
for the rewrite history.

## Synthetic benchmark dataset

`data/synthetic_tracks.csv` (10,500 rows) and `data/synthetic_video_segments.csv`
(1,021 rows) are produced by `scripts/generate_synthetic_dataset.py`.
Audio feature distributions are calibrated against publicly reported
Spotify aggregates across 12 genre profiles. Every row carries
`_synthetic=True` so the data origin is unambiguous.

The synthetic dataset exists because Spotify's audio-features
deprecation makes pulling 10K real audio features impossible for new
apps. The pipeline code is identical for synthetic and real data — the
only difference is what gets handed to it. To regenerate:

```bash
python3 scripts/generate_synthetic_dataset.py --tracks 10500 --segments 1100
```

## Setup for live operation

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

`GOOGLE_APPLICATION_CREDENTIALS` and `BUCKET_NAME` are only needed for
`--analyze-video`. Spotify-only operations work without them.

## Usage

```bash
python main.py --fetch-spotify                 # build metadata + features xlsx
python main.py --analyze-video                 # run Video Intelligence on bucket
python main.py --train-model                   # train classifier on real features
python main.py --recommend                     # rank tracks for the analyzed video
python main.py --full-pipeline                 # run all four in order
python main.py --fetch-spotify --playlist-id <ID>
```

## Repository layout

| Path | Purpose |
| --- | --- |
| `main.py` | CLI orchestration, lazy-init for both API clients |
| `audio_features.py` | Spotify API → librosa fallback feature extractor |
| `recommend_spotify_playlist_music_for_tiktok_edits.py` | Playlist fetch with feature attachment |
| `AutoLabel.py` | `MusicEmotionClassifier` (Keras MLP, real features) |
| `GoogleVideoIntelligenceAPI.py` | GCS + Video Intelligence wrapper, lazy clients |
| `scripts/generate_synthetic_dataset.py` | Distribution-matched synthetic dataset generator |
| `scripts/evaluate.py` | Train + held-out evaluation, writes results/ |
| `scripts/benchmark.py` | Latency + processing time benchmarks |
| `data/synthetic_tracks.csv` | 10,500-row synthetic benchmark dataset |
| `data/synthetic_video_segments.csv` | 1,021-row synthetic video segment dataset |
| `results/evaluation.json` | Accuracy + per-class metrics |
| `results/benchmark.json` | Latency + processing time numbers |
| `results/confusion_matrix.png` | Held-out confusion matrix visualization |
| `docs/EVALUATION.md` | Detailed evaluation methodology and limits |
| `docs/EDGE_CASES.md` | Documented edge cases and their handlers |
| `docs/ITERATION_LOG.md` | Iteration history with measured outcomes |

## License

MIT — see [LICENSE](LICENSE).
