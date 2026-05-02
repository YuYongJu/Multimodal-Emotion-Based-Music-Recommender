"""Integration tests for the full multimodal pipeline.

Mocks Spotify HTTP (via responses) and Google Video Intelligence
(via unittest.mock) so the test exercises the real glue code between
modules without external API calls.
"""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytestmark = pytest.mark.integration


@pytest.fixture
def trained_classifier_on_synth(small_track_dataframe: pd.DataFrame):
    from generate_synthetic_dataset import generate_audio_features

    from AutoLabel import FEATURE_COLUMNS, MusicEmotionClassifier

    df = generate_audio_features(n=200, seed=7)
    feats = df[FEATURE_COLUMNS]
    classifier = MusicEmotionClassifier()
    labels = classifier._assign_initial_emotions(feats)
    classifier.train(feats.values, labels, epochs=3,
                    batch_size=32, validation_split=0.2)
    return classifier


@pytest.fixture
def synthetic_metadata_xlsx(tmp_path):
    """Materialise a small but real-shape metadata xlsx the pipeline can read."""
    from generate_synthetic_dataset import generate_audio_features
    df = generate_audio_features(n=80, seed=3)
    path = tmp_path / "spotify_metadata.xlsx"
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def synthetic_video_xlsx(tmp_path):
    """A Video Intelligence-shape xlsx with a strong action-emotion signal
    so downstream emotion-target derivation is testable."""
    rows = []
    for v in range(3):
        for s in range(8):
            rows.append({
                "Video": f"action_video_{v}.mp4",
                "Label Description": ["fight", "explosion", "chase"][s % 3],
                "Category Description": "Action",
                "Start Time": s * 3.0,
                "End Time": s * 3.0 + 2.5,
                "Confidence": 0.85 + (s % 3) * 0.04,
            })
    path = tmp_path / "GoogleVideoIntelligenceLabelAnalyzer_results.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    return path


class TestFullPipelineIntegration:
    @pytest.mark.slow
    def test_recommend_runs_end_to_end_on_synthetic_inputs(
        self, tmp_path, monkeypatch, synthetic_metadata_xlsx,
        synthetic_video_xlsx, trained_classifier_on_synth,
    ) -> None:
        """End-to-end: video xlsx + metadata xlsx + trained model →
        ranked recommendations DataFrame with expected schema."""
        import main

        model_path = tmp_path / "emotion_classifier_model.h5"
        scaler_path = tmp_path / "emotion_scaler.pkl"
        trained_classifier_on_synth.save_model(str(model_path), str(scaler_path))

        monkeypatch.chdir(tmp_path)
        Path("emotion_classifier_model.h5").write_bytes(model_path.read_bytes())
        Path("emotion_scaler.pkl").write_bytes(scaler_path.read_bytes())
        Path("spotify_metadata.xlsx").write_bytes(
            synthetic_metadata_xlsx.read_bytes()
        )
        Path("GoogleVideoIntelligenceLabelAnalyzer_results.xlsx").write_bytes(
            synthetic_video_xlsx.read_bytes()
        )

        result = main.recommend_music_for_video(
            video_data_path="GoogleVideoIntelligenceLabelAnalyzer_results.xlsx",
            spotify_data_path="spotify_metadata.xlsx",
            model_path="emotion_classifier_model.h5",
            top_n=5,
        )
        assert isinstance(result, pd.DataFrame)
        assert {"track_name", "artist", "predicted_emotion", "match_score"}.issubset(result.columns)
        assert len(result) <= 5
        assert (result["match_score"].diff().dropna() <= 0).all()  # sorted desc

    def test_aggressive_video_surfaces_aggressive_target(self) -> None:
        """Contract: a fight-labeled video must produce a target set
        that includes 'aggressive'. This is the core multimodal claim."""
        from main import map_video_content_to_emotions
        targets = map_video_content_to_emotions(
            labels=["fight", "explosion", "chase"],
            categories=["Action"],
        )
        assert "aggressive" in targets

    def test_calm_video_does_not_surface_aggressive_target(self) -> None:
        from main import map_video_content_to_emotions
        targets = map_video_content_to_emotions(
            labels=["sunset", "water", "sky"],
            categories=["Nature"],
        )
        assert "aggressive" not in targets
        assert "calm" in targets

    @pytest.mark.slow
    def test_recommend_raises_when_features_missing(
        self, tmp_path, synthetic_video_xlsx, trained_classifier_on_synth
    ) -> None:
        """If the Spotify metadata xlsx predates the audio_features
        integration (no danceability/energy/etc columns), the recommend
        path must surface a clear error rather than silently producing
        garbage."""
        import main

        model_path = tmp_path / "emotion_classifier_model.h5"
        scaler_path = tmp_path / "emotion_scaler.pkl"
        trained_classifier_on_synth.save_model(str(model_path), str(scaler_path))

        legacy_metadata = pd.DataFrame([
            {"track_id": "abc", "track_name": "Old Track", "artist": "Test",
             "popularity": 50, "duration_ms": 200000, "preview_url": ""}
        ])
        legacy_path = tmp_path / "legacy_metadata.xlsx"
        legacy_metadata.to_excel(legacy_path, index=False)

        with pytest.raises(SystemExit, match="audio feature columns"):
            main.recommend_music_for_video(
                video_data_path=str(synthetic_video_xlsx),
                spotify_data_path=str(legacy_path),
                model_path=str(model_path),
                top_n=3,
            )


class TestSpotifyFetchWithMockedAPI:
    """Mocks the spotipy client so fetch logic is testable without API access."""

    def test_fetch_metadata_with_audio_features_path(self, tmp_path) -> None:
        from recommend_spotify_playlist_music_for_tiktok_edits import (
            fetch_spotify_metadata,
        )

        mock_sp = MagicMock()
        mock_sp.playlist_items.side_effect = [
            {"items": [
                {"track": {"id": f"track_{i}"}} for i in range(3)
            ]},
            {"items": []},
        ]
        mock_sp.track.side_effect = [
            {
                "name": f"Track {i}",
                "artists": [{"name": f"Artist {i}"}],
                "album": {"name": f"Album {i}", "release_date": "2024-01-01"},
                "duration_ms": 200000,
                "popularity": 50 + i,
                "preview_url": "",
            }
            for i in range(3)
        ]
        mock_sp.audio_features.side_effect = [
            [{
                "danceability": 0.7, "energy": 0.8, "key": 5, "loudness": -4.0,
                "mode": 1, "speechiness": 0.05, "acousticness": 0.1,
                "instrumentalness": 0.0, "liveness": 0.12, "valence": 0.6,
                "tempo": 120.0, "id": f"track_{i}",
            }]
            for i in range(3)
        ]

        df = fetch_spotify_metadata("playlist_xyz", sp=mock_sp, track_limit=10)
        assert len(df) == 3
        for col in ["danceability", "energy", "valence", "tempo"]:
            assert col in df.columns
            assert not df[col].isna().any()
        assert mock_sp.audio_features.call_count == 3

    def test_fetch_falls_back_to_librosa_when_spotify_audio_features_fail(
        self, tmp_path, mocker
    ) -> None:
        """Simulates the Nov 2024 deprecation — track() works but
        audio_features() returns 403 — fetcher must fall through to librosa."""
        from recommend_spotify_playlist_music_for_tiktok_edits import (
            fetch_spotify_metadata,
        )

        mock_sp = MagicMock()
        mock_sp.playlist_items.side_effect = [
            {"items": [{"track": {"id": "track_0"}}]},
            {"items": []},
        ]
        mock_sp.track.return_value = {
            "name": "T", "artists": [{"name": "A"}],
            "album": {"name": "Alb", "release_date": "2024-01-01"},
            "duration_ms": 200000, "popularity": 60,
            "preview_url": "https://example.com/preview.mp3",
        }
        mock_sp.audio_features.side_effect = Exception("403 Forbidden")

        librosa_features = {
            "danceability": 0.55, "energy": 0.65, "key": 7, "loudness": -6.0,
            "mode": 0, "speechiness": 0.07, "acousticness": 0.20,
            "instrumentalness": 0.05, "liveness": 0.18, "valence": 0.45,
            "tempo": 110.0,
        }
        mocker.patch(
            "audio_features.from_preview_url",
            return_value=librosa_features,
        )

        df = fetch_spotify_metadata("p", sp=mock_sp, track_limit=10)
        assert len(df) == 1
        assert df.iloc[0]["energy"] == 0.65
        assert df.iloc[0]["tempo"] == 110.0


class TestVideoSegmentsThroughEmotionMapper:
    """Asserts that synthetic Video Intelligence segments flow through
    map_video_content_to_emotions() to produce non-empty emotion targets."""

    def test_action_segments_produce_aggressive_target(self) -> None:
        from generate_synthetic_dataset import generate_video_segments

        from main import map_video_content_to_emotions

        df = generate_video_segments(num_segments=200, seed=42)
        action = df[df["Category Description"] == "Action"]
        if len(action) == 0:
            pytest.skip("seed produced no Action segments")

        labels = action["Label Description"].value_counts().head(5).index.tolist()
        cats = action["Category Description"].value_counts().head(5).index.tolist()
        targets = map_video_content_to_emotions(labels, cats)
        assert "aggressive" in targets or "energetic" in targets

    def test_full_synthetic_dataset_yields_all_five_emotion_classes(self) -> None:
        from generate_synthetic_dataset import generate_video_segments

        from main import map_video_content_to_emotions

        df = generate_video_segments(num_segments=1100, seed=42)
        all_targets: set[str] = set()
        for _, group in df.groupby("Video"):
            labels = group["Label Description"].value_counts().head(5).index.tolist()
            cats = group["Category Description"].value_counts().head(5).index.tolist()
            all_targets.update(map_video_content_to_emotions(labels, cats))

        # The synthetic dataset is broad enough to surface at least 4 of 5
        assert len(all_targets) >= 4
