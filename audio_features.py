"""Audio feature extraction with two real sources and no fabricated data.

Spotify deprecated the public audio-features endpoint for new third-party
apps on 2024-11-27. Apps registered before then may still have access, so
we try the Spotify API first and fall back to local DSP via librosa on the
30-second preview URL when the API is unavailable.

Returned features are aligned with the Spotify schema so downstream code
(MusicEmotionClassifier) sees the same column names regardless of source.
"""
from __future__ import annotations

import io
import math
import os
from typing import Optional

import numpy as np
import requests

FEATURE_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo",
]

_REQUEST_TIMEOUT = 15


class FeatureUnavailable(RuntimeError):
    pass


def from_spotify(sp, track_id: str) -> Optional[dict]:
    """Try the Spotify Web API audio-features endpoint."""
    try:
        result = sp.audio_features([track_id])
    except Exception:
        return None
    if not result or result[0] is None:
        return None
    feat = result[0]
    return {col: feat.get(col) for col in FEATURE_COLUMNS}


def from_preview_url(preview_url: str) -> Optional[dict]:
    """Compute Spotify-shaped features from the 30-second preview MP3.

    Uses librosa for the DSP. The mapping to Spotify's proprietary
    danceability/valence is a heuristic — these are not the same numbers
    Spotify would return, but they are derived from real audio rather than
    fabricated, and they preserve the relative ordering the classifier
    needs (high-energy tracks get higher energy scores, etc.).
    """
    if not preview_url:
        return None
    try:
        import librosa
    except ImportError:
        return None

    try:
        response = requests.get(preview_url, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        audio_bytes = io.BytesIO(response.content)
        y, sr = librosa.load(audio_bytes, sr=22050, mono=True)
    except Exception:
        return None

    if y.size == 0:
        return None

    rms = float(np.sqrt(np.mean(y ** 2)))
    energy = float(np.clip(rms * 4.0, 0.0, 1.0))

    rms_db = 20 * math.log10(rms + 1e-9)
    loudness = float(np.clip(rms_db, -60.0, 0.0))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.clip(tempo if tempo else 120.0, 50.0, 200.0))

    onset_strength_norm = float(np.clip(np.mean(onset_env) / 5.0, 0.0, 1.0))
    danceability = float((onset_strength_norm + min(tempo / 180.0, 1.0)) / 2.0)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = int(np.argmax(np.mean(chroma, axis=1)))
    chroma_per_pc = np.mean(chroma, axis=1)
    major_template = np.array([1, 0, 0.5, 0, 1, 0.5, 0, 1, 0, 0.5, 0, 0.5])
    minor_template = np.array([1, 0, 0.5, 1, 0, 0.5, 0, 1, 0.5, 0, 0.5, 0])
    rolled = np.roll(chroma_per_pc, -key)
    mode = 1 if np.dot(rolled, major_template) >= np.dot(rolled, minor_template) else 0

    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    valence = float(np.clip(spectral_centroid / 4000.0, 0.0, 1.0))

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    speechiness = float(np.clip(zcr * 4.0, 0.0, 1.0))

    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    instrumentalness = float(np.clip(flatness * 5.0, 0.0, 1.0))
    acousticness = float(np.clip(1.0 - energy + 0.1, 0.0, 1.0))

    rms_var = float(np.var(librosa.feature.rms(y=y)))
    liveness = float(np.clip(rms_var * 50.0, 0.0, 1.0))

    return {
        "danceability": danceability,
        "energy": energy,
        "key": key,
        "loudness": loudness,
        "mode": mode,
        "speechiness": speechiness,
        "acousticness": acousticness,
        "instrumentalness": instrumentalness,
        "liveness": liveness,
        "valence": valence,
        "tempo": tempo,
    }


def get_audio_features(track_id: str, preview_url: Optional[str], sp=None) -> dict:
    """Get audio features for a track using whichever source is available.

    Resolution order: Spotify API → librosa on preview URL → FeatureUnavailable.
    Tracks with no preview_url and no Spotify API access cannot be classified
    with this system; the caller should drop them or surface the gap to the
    user rather than substitute fake values.
    """
    if sp is not None:
        feat = from_spotify(sp, track_id)
        if feat is not None:
            feat["_source"] = "spotify_api"
            return feat

    if preview_url:
        feat = from_preview_url(preview_url)
        if feat is not None:
            feat["_source"] = "librosa_preview"
            return feat

    raise FeatureUnavailable(
        f"No audio features available for track {track_id}: "
        "Spotify API returned no result and no preview_url is available "
        "for librosa fallback."
    )
