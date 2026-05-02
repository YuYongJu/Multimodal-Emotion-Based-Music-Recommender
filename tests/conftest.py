"""Shared pytest fixtures for the music-emotion test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


@pytest.fixture(scope="session")
def feature_columns() -> list[str]:
    from audio_features import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


@pytest.fixture
def small_track_dataframe(feature_columns: list[str]) -> pd.DataFrame:
    """A 50-row Spotify-shape DataFrame with deterministic features.

    Spans the full feature space (low-energy/calm tracks at the start,
    high-energy/aggressive at the end) so emotion-classification tests
    can pick rows of known character without random sampling.
    """
    rng = np.random.default_rng(0)
    n = 50
    energy = np.linspace(0.05, 0.98, n)
    valence = np.linspace(0.10, 0.95, n)
    rows = []
    for i in range(n):
        rows.append(
            {
                "track_id": f"test_{i:03d}",
                "track_name": f"Test Track {i}",
                "artist": f"Test Artist {i // 5}",
                "album_name": f"Test Album {i // 3}",
                "release_date": "2024-01-01",
                "duration_ms": 200000 + i * 1000,
                "popularity": int(rng.integers(20, 80)),
                "preview_url": "",
                "danceability": float(np.clip(0.3 + energy[i] * 0.5, 0, 1)),
                "energy": float(energy[i]),
                "key": int(i % 12),
                "loudness": float(np.clip(-20 + energy[i] * 18, -60, 0)),
                "mode": int(i % 2),
                "speechiness": float(rng.uniform(0.0, 0.2)),
                "acousticness": float(np.clip(1 - energy[i] + rng.uniform(-0.1, 0.1), 0, 1)),
                "instrumentalness": float(rng.uniform(0.0, 0.3)),
                "liveness": float(rng.uniform(0.05, 0.4)),
                "valence": float(valence[i]),
                "tempo": float(np.clip(80 + energy[i] * 60, 50, 200)),
                "_synthetic": True,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def video_segments_dataframe() -> pd.DataFrame:
    """A small Video Intelligence-shape DataFrame across two videos."""
    rows = [
        {
            "Video": "v1.mp4",
            "Label Description": "fight",
            "Category Description": "Action",
            "Start Time": 0.0,
            "End Time": 3.0,
            "Confidence": 0.92,
        },
        {
            "Video": "v1.mp4",
            "Label Description": "explosion",
            "Category Description": "Action",
            "Start Time": 3.0,
            "End Time": 5.5,
            "Confidence": 0.87,
        },
        {
            "Video": "v1.mp4",
            "Label Description": "chase",
            "Category Description": "Action",
            "Start Time": 5.5,
            "End Time": 9.0,
            "Confidence": 0.81,
        },
        {
            "Video": "v2.mp4",
            "Label Description": "sunset",
            "Category Description": "Nature",
            "Start Time": 0.0,
            "End Time": 4.0,
            "Confidence": 0.95,
        },
        {
            "Video": "v2.mp4",
            "Label Description": "water",
            "Category Description": "Nature",
            "Start Time": 4.0,
            "End Time": 7.5,
            "Confidence": 0.91,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def silent_audio_buffer() -> bytes:
    """A trivially small WAV buffer of silence for librosa fallback tests.

    Two channels of zero samples at 22.05 kHz for 0.5 s. Round-trip
    through soundfile produces a parseable but feature-impoverished
    audio buffer — exercising librosa's behavior on near-empty input.
    """
    import io

    import soundfile as sf

    sample_rate = 22050
    duration_s = 0.5
    silent = np.zeros(int(sample_rate * duration_s), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, silent, sample_rate, format="WAV")
    return buf.getvalue()
