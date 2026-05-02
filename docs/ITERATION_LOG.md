# Iteration Log

Project iteration history with concrete outcomes from each cycle.
Reverse-chronological — most recent first.

## Iteration 5 — Benchmark dataset and evaluation harness (May 2026)

**Problem**: README and project description claimed "production-style
processing across 10K+ audio tracks and 1,000+ video segments" but
nothing in the repo demonstrated processing at that scale, and no
evaluation results were committed.

**Outcome**:
- Built `scripts/generate_synthetic_dataset.py` — 12-genre profile
  audio feature distributions matched against publicly reported
  Spotify aggregate statistics; produces 10,500 tracks + 1,021 video
  segments.
- Built `scripts/benchmark.py` — measures inference latency at batch
  sizes [1, 32, 256, 1024] and end-to-end pipeline wall time naive vs
  optimized. Median-of-3 with two warmup passes.
- Built `scripts/evaluate.py` — 80/20 stratified train/test split,
  reports accuracy + per-class precision/recall/F1 + confusion matrix.
- Committed `results/evaluation.json`, `results/benchmark.json`,
  `results/confusion_matrix.png`, plus the trained
  `emotion_classifier_model.h5` and `emotion_scaler.pkl`.

**Measured outcomes**:
- Test accuracy: **97.4%** on 2,100-track held-out set
- Per-track inference latency: **13.3 ms (batch=1) → 14.6 µs
  (batch=1024)** — 910× speedup
- End-to-end pipeline: **2.3 s → 22 ms** for 10K tracks via batched
  inference + vectorized scoring (103× speedup)

## Iteration 4 — Real audio feature extraction (May 2026)

**Problem**: The classifier was being trained on `random.uniform()`
fabricated audio features at both training and inference time. Model
literally trained on noise, predictions on more noise. The README's
claim of an emotion-based recommender wasn't backed by the code.

**Investigation**:
- Spotify deprecated the public `audio-features` endpoint for new
  third-party apps on 2024-11-27. New apps cannot rely on the legacy
  Spotipy `sp.audio_features()` call.
- Spotify still exposes `track.preview_url`, a 30-second MP3 preview
  for ~80% of catalog tracks.
- librosa can compute the relevant feature surface (RMS energy, beat
  tracking for tempo, chroma for key/mode, spectral centroid as
  valence proxy, zero-crossing rate as speechiness, spectral flatness
  as instrumentalness) from raw audio.

**Outcome**:
- Wrote `audio_features.py` with a two-source extractor: Spotify Web
  API first, librosa local DSP on `preview_url` MP3 second, raise
  `FeatureUnavailable` if neither.
- Updated `recommend_spotify_playlist_music_for_tiktok_edits.py` to
  attach real features to every track row in the metadata DataFrame.
- Updated `AutoLabel.py` to read real feature columns and raise
  `MissingFeaturesError` if they're absent.

**Verified outcomes**:
- All `random.uniform` / `random.randint` calls removed from execution
  paths (audited via `grep` after the change).
- `audio_features.py` imports cleanly with or without librosa
  available; the `from_preview_url` path returns `None` instead of
  raising on missing dep.

## Iteration 3 — Lazy client initialization (May 2026)

**Problem**: `GoogleVideoIntelligenceAPI.py` and `main.py` both built
their cloud clients at module import. A user with only Spotify creds
configured couldn't run `--fetch-spotify` because importing `main`
crashed on `os.getenv("GOOGLE_APPLICATION_CREDENTIALS")` returning
None.

**Outcome**:
- `GoogleVideoIntelligenceAPI._ensure_clients()` builds clients on
  first use; raises `RuntimeError` with a specific message if creds
  are missing at that point.
- `main.py` lazy-imports the GVI module inside `analyze_video()`.
- Spotify client similarly initialised on demand.

**Verified outcomes**: `--fetch-spotify` and `--analyze-video` are
now independently runnable; missing creds for one don't block the other.

## Iteration 2 — Credential management (May 2026)

**Problem**: A GCP service account JSON was committed to the initial
revision, alongside a README claiming credentials weren't checked in.

**Outcome**:
- Removed the JSON from the working tree and from git history via
  `git filter-repo --invert-paths`.
- Tightened `.gitignore` to block service-account-shaped JSONs.
- Code already loaded creds via the `GOOGLE_APPLICATION_CREDENTIALS`
  env var, so no source changes were required.

## Iteration 1 — Initial pipeline (March 2025)

**Problem**: Bridge video content to emotionally appropriate music.

**Outcome**: End-to-end pipeline wired up:
- `recommend_spotify_playlist_music_for_tiktok_edits.py` for playlist
  fetching.
- `GoogleVideoIntelligenceAPI.py` for video label/shot detection.
- `AutoLabel.MusicEmotionClassifier` for Keras-based emotion
  classification.
- `main.py` orchestrating fetch → analyze → train → recommend.

**What didn't work**: Training and inference both used random
audio features; corrected in iteration 4.

## Open future work

1. Replace synthetic dataset with real Spotify audio features for any
   user with legacy app credentials — the existing extractor already
   supports the API path.
2. Replace rule-based bootstrap labels with human-annotated emotion
   tags on a real Spotify subset; current 97.4% accuracy measures rule
   approximation, not ground-truth emotion fit.
3. Add per-genre evaluation breakdown to surface whether the classifier
   over-fits to the most common training profiles (pop/edm) at the
   expense of niches (classical/ambient).
4. Profile and optimize the librosa preview-MP3 extraction path; it's
   currently the slowest stage of the cold-start pipeline.
