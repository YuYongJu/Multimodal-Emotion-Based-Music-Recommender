# Edge Cases

Documented edge cases observed and handled across the recommender
pipeline. Each entry names the failure mode, the code path that catches
it, and what the user sees when it triggers.

## Audio feature acquisition

### 1. Spotify `audio-features` endpoint deprecated for new apps
**Symptom**: HTTP 403 / 404 from `sp.audio_features([track_id])` for
fresh-registered Spotify apps after 2024-11-27.
**Where caught**: `audio_features.from_spotify()` returns `None` on any
exception or empty result, and the caller falls through to the librosa
preview-MP3 path.
**User-visible effect**: The fetch script prints
`librosa fallback used for N tracks` in its summary and proceeds.

### 2. Track has no `preview_url`
**Symptom**: ~20% of tracks Spotify returns expose `preview_url = null`,
typically due to label restrictions, regional availability, or podcast
content.
**Where caught**: `audio_features.get_audio_features()` raises
`FeatureUnavailable` with a specific message; the fetch loop catches it
and skips the track rather than emitting NaN features.
**User-visible effect**: Skipped track count in the fetch summary.

### 3. librosa beat-track returns 0
**Symptom**: For audio with no clear pulse (ambient, spoken-word,
heavily textured), `librosa.beat.beat_track` can return `tempo=0`.
**Where caught**: `audio_features.from_preview_url()` clips with
`np.clip(tempo if tempo else 120.0, 50.0, 200.0)` so any zero or
out-of-range value defaults to 120 BPM.
**Trade-off**: The fallback default is a guess, but feeding 0 BPM into
the danceability heuristic produces NaN propagation. Worth the bias.

### 4. librosa import unavailable
**Symptom**: `from_preview_url()` called in an environment without
librosa installed.
**Where caught**: A guarded `import librosa` returns `None` instead of
raising `ImportError`, then `get_audio_features()` raises
`FeatureUnavailable` so the caller can decide whether to skip.

## Network and credentials

### 5. Spotify rate limits (HTTP 429)
**Symptom**: Bursts of `sp.playlist_items()` calls hit Spotify's
client-credentials rate limit (~1 req/sec sustained).
**Where caught**: spotipy's built-in retry wrapper handles transient
429s with exponential backoff. Our code adds a `track_limit=50` default
to keep playlist fetches under the burst threshold.

### 6. Missing `SPOTIFY_CLIENT_ID` or `SPOTIFY_CLIENT_SECRET`
**Symptom**: A user runs `--fetch-spotify` without setting the env vars.
**Where caught**: `_get_spotify_client()` raises `RuntimeError` with a
specific message before any network call.

### 7. Missing `GOOGLE_APPLICATION_CREDENTIALS`
**Symptom**: A user runs `--analyze-video` without GCP creds.
**Where caught**: `GoogleVideoIntelligenceAPI._ensure_clients()` raises
`RuntimeError` and the import-time path no longer crashes the whole
module — `--fetch-spotify` still works in this state.

## Classifier training and inference

### 8. Metadata file predates `audio_features` integration
**Symptom**: Old `spotify_metadata.xlsx` files have `track_id`,
`track_name`, `artist`, etc. but no `danceability`/`energy`/etc.
**Where caught**: `MusicEmotionClassifier._features_from_dataframe()`
raises `MissingFeaturesError` listing the missing columns and
suggesting `--fetch-spotify`.

### 9. NaN audio features after extraction
**Symptom**: librosa silence input or download truncation produces NaN
in some columns.
**Where caught**: `preprocess_data()` calls `df.dropna()` and raises
`MissingFeaturesError` if every row has at least one NaN.

### 10. Empty playlist or all-track-skip outcome
**Symptom**: `fetch_spotify_metadata()` returns an empty DataFrame.
**Where caught**: Caller in `main.fetch_spotify_data()` raises
`SystemExit` with a clear message rather than writing an empty xlsx.

### 11. Missing trained model on `--recommend`
**Symptom**: User runs `--recommend` without `--train-model` first.
**Where caught**: `recommend_music_for_video()` checks for the `.h5`
file; if missing, it transparently calls `train_emotion_classifier()`
on the fetched Spotify data before continuing. No crash.

### 12. Video analysis output schema mismatch
**Symptom**: Older Video Intelligence output xlsx files might not have
`Confidence` or `Category Description` columns.
**Where caught**: `recommend_music_for_video()` reads the xlsx and the
column-access calls would raise `KeyError`. **Currently not handled** —
relies on schema consistency from `analyze_videos_in_bucket()`.

## Synthetic dataset

### 13. Synthetic data accidentally treated as real
**Symptom**: A user re-runs the system on `data/synthetic_tracks.csv`
and assumes the predictions are about real Spotify catalog entries.
**Where caught**: Every synthetic row carries `_synthetic=True`; the
README and `docs/EVALUATION.md` lead with the synthetic-data
disclosure. Not enforced in code — documentation-level only.
