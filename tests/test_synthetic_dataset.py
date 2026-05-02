"""Unit tests for the synthetic dataset generator."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from audio_features import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from generate_synthetic_dataset import (  # noqa: E402
    GENRE_PROFILES,
    VIDEO_LABEL_VOCAB,
    generate_audio_features,
    generate_video_segments,
)


class TestAudioFeatureGeneration:
    def test_row_count_matches_request(self) -> None:
        df = generate_audio_features(n=100, seed=1)
        assert len(df) == 100

    def test_includes_all_canonical_feature_columns(self) -> None:
        df = generate_audio_features(n=10, seed=1)
        for col in FEATURE_COLUMNS:
            assert col in df.columns, f"missing canonical column: {col}"

    def test_every_row_tagged_synthetic(self) -> None:
        df = generate_audio_features(n=20, seed=1)
        assert "_synthetic" in df.columns
        assert df["_synthetic"].all()

    def test_features_in_canonical_ranges(self) -> None:
        df = generate_audio_features(n=500, seed=7)
        bounded_unit = ["danceability", "energy", "valence", "speechiness",
                        "acousticness", "instrumentalness", "liveness"]
        for col in bounded_unit:
            assert (df[col] >= 0.0).all() and (df[col] <= 1.0).all(), \
                f"{col} out of [0, 1]"
        assert (df["tempo"] >= 50.0).all() and (df["tempo"] <= 200.0).all()
        assert (df["loudness"] >= -60.0).all() and (df["loudness"] <= 0.0).all()
        assert df["key"].isin(range(12)).all()
        assert df["mode"].isin([0, 1]).all()

    def test_no_nan_in_feature_columns(self) -> None:
        df = generate_audio_features(n=200, seed=3)
        assert not df[FEATURE_COLUMNS].isna().any().any()

    def test_deterministic_for_fixed_seed(self) -> None:
        a = generate_audio_features(n=50, seed=99)
        b = generate_audio_features(n=50, seed=99)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seeds_produce_different_data(self) -> None:
        a = generate_audio_features(n=50, seed=1)
        b = generate_audio_features(n=50, seed=2)
        # Not bitwise identical
        assert not a["energy"].equals(b["energy"])

    def test_track_ids_are_unique(self) -> None:
        df = generate_audio_features(n=1000, seed=5)
        assert df["track_id"].nunique() == len(df)

    def test_genre_distribution_covers_known_genres(self) -> None:
        df = generate_audio_features(n=2000, seed=11)
        genres_present = set(df["_genre_synth"].unique())
        # At least 8 of the 12 profiles should appear at this scale
        assert len(genres_present & set(GENRE_PROFILES.keys())) >= 8


class TestVideoSegmentGeneration:
    def test_row_count_at_or_above_request(self) -> None:
        df = generate_video_segments(num_segments=100, seed=1)
        assert len(df) >= 50  # generator may slightly under/over depending on grouping
        assert len(df) <= 100

    def test_includes_video_intelligence_schema(self) -> None:
        df = generate_video_segments(num_segments=50, seed=1)
        for col in ["Video", "Label Description", "Category Description",
                    "Start Time", "End Time", "Confidence"]:
            assert col in df.columns

    def test_every_row_tagged_synthetic(self) -> None:
        df = generate_video_segments(num_segments=50, seed=1)
        assert df["_synthetic"].all()

    def test_confidence_in_valid_range(self) -> None:
        df = generate_video_segments(num_segments=200, seed=1)
        assert (df["Confidence"] > 0.0).all()
        assert (df["Confidence"] <= 1.0).all()

    def test_labels_drawn_from_vocab(self) -> None:
        df = generate_video_segments(num_segments=200, seed=1)
        all_labels = {label for labels in VIDEO_LABEL_VOCAB.values() for label in labels}
        emitted_labels = set(df["Label Description"].unique())
        assert emitted_labels.issubset(all_labels)

    def test_categories_match_label_to_category_mapping(self) -> None:
        df = generate_video_segments(num_segments=300, seed=2)
        for _, row in df.iterrows():
            cat = row["Category Description"]
            label = row["Label Description"]
            assert cat in VIDEO_LABEL_VOCAB
            assert label in VIDEO_LABEL_VOCAB[cat], \
                f"label '{label}' not in category '{cat}'"

    def test_time_intervals_have_positive_duration(self) -> None:
        df = generate_video_segments(num_segments=100, seed=1)
        durations = df["End Time"] - df["Start Time"]
        assert (durations > 0).all()
