"""Dataset sanity tests for the committed synthetic benchmark.

These run against the actual files in data/ so a future change that
silently corrupts the dataset (e.g., regenerating without _synthetic
flags, dropping feature columns, drifting distributions) fails CI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from audio_features import FEATURE_COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[1]
TRACKS_PATH = REPO_ROOT / "data" / "synthetic_tracks.csv"
SEGMENTS_PATH = REPO_ROOT / "data" / "synthetic_video_segments.csv"


@pytest.fixture(scope="module")
def tracks_df() -> pd.DataFrame:
    if not TRACKS_PATH.exists():
        pytest.skip("synthetic_tracks.csv not committed")
    return pd.read_csv(TRACKS_PATH)


@pytest.fixture(scope="module")
def segments_df() -> pd.DataFrame:
    if not SEGMENTS_PATH.exists():
        pytest.skip("synthetic_video_segments.csv not committed")
    return pd.read_csv(SEGMENTS_PATH)


class TestSchemaContract:
    def test_all_canonical_audio_features_present(self, tracks_df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLUMNS if c not in tracks_df.columns]
        assert missing == [], f"missing canonical columns: {missing}"

    def test_track_metadata_columns_present(self, tracks_df: pd.DataFrame) -> None:
        for col in ("track_id", "track_name", "artist", "album_name", "popularity", "duration_ms"):
            assert col in tracks_df.columns

    def test_synthetic_flag_present(self, tracks_df: pd.DataFrame) -> None:
        assert "_synthetic" in tracks_df.columns

    def test_video_intelligence_schema(self, segments_df: pd.DataFrame) -> None:
        for col in (
            "Video",
            "Label Description",
            "Category Description",
            "Start Time",
            "End Time",
            "Confidence",
        ):
            assert col in segments_df.columns


class TestProvenanceInvariants:
    """Every committed row must be tagged synthetic — protects against
    accidentally committing real Spotify track data with the synthetic
    dataset, which would be a privacy/licensing concern."""

    def test_every_track_is_synthetic(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["_synthetic"].all(), (
            "non-synthetic rows committed to data/synthetic_tracks.csv"
        )

    def test_every_segment_is_synthetic(self, segments_df: pd.DataFrame) -> None:
        assert segments_df["_synthetic"].all()

    def test_no_real_spotify_track_ids(self, tracks_df: pd.DataFrame) -> None:
        """Real Spotify track IDs are 22-char base62. Synthetic IDs use
        the 'syn_NNNNNN' prefix. A drift here means real data leaked in."""
        assert tracks_df["track_id"].str.startswith("syn_").all()


class TestScaleInvariants:
    """The repo's headline numbers depend on the dataset being at or
    above 10K tracks and 1K segments. These tests catch silent
    truncation."""

    def test_track_count_at_least_10k(self, tracks_df: pd.DataFrame) -> None:
        assert len(tracks_df) >= 10_000, (
            f"resume bullet claims '10K+ tracks' but dataset has {len(tracks_df)}"
        )

    def test_segment_count_at_least_1k(self, segments_df: pd.DataFrame) -> None:
        assert len(segments_df) >= 1_000, (
            f"resume bullet claims '1K+ segments' but dataset has {len(segments_df)}"
        )

    def test_segments_span_at_least_25_videos(self, segments_df: pd.DataFrame) -> None:
        n_videos = segments_df["Video"].nunique()
        assert n_videos >= 25, f"only {n_videos} unique videos in segment dataset"


class TestNoNaNs:
    def test_no_nan_in_audio_features(self, tracks_df: pd.DataFrame) -> None:
        nan_counts = tracks_df[FEATURE_COLUMNS].isna().sum()
        offenders = nan_counts[nan_counts > 0]
        assert offenders.empty, f"NaN in audio features: {offenders.to_dict()}"

    def test_no_nan_in_segment_confidence(self, segments_df: pd.DataFrame) -> None:
        assert not segments_df["Confidence"].isna().any()

    def test_no_nan_in_segment_labels(self, segments_df: pd.DataFrame) -> None:
        assert not segments_df["Label Description"].isna().any()
        assert not segments_df["Category Description"].isna().any()


class TestValueRanges:
    """Catch distribution drift — if a future regenerator change
    accidentally widens a feature's range, this fails fast."""

    def test_unit_bounded_features(self, tracks_df: pd.DataFrame) -> None:
        unit = [
            "danceability",
            "energy",
            "valence",
            "speechiness",
            "acousticness",
            "instrumentalness",
            "liveness",
        ]
        for col in unit:
            assert tracks_df[col].min() >= 0.0, f"{col} has negatives"
            assert tracks_df[col].max() <= 1.0, f"{col} exceeds 1.0"

    def test_tempo_in_natural_range(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["tempo"].min() >= 50.0
        assert tracks_df["tempo"].max() <= 200.0

    def test_loudness_in_db_range(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["loudness"].min() >= -60.0
        assert tracks_df["loudness"].max() <= 0.0

    def test_key_in_chromatic_range(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["key"].isin(range(12)).all()

    def test_mode_is_binary(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["mode"].isin([0, 1]).all()

    def test_popularity_in_zero_hundred(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["popularity"].min() >= 0
        assert tracks_df["popularity"].max() <= 100

    def test_segment_intervals_have_positive_duration(self, segments_df: pd.DataFrame) -> None:
        durations = segments_df["End Time"] - segments_df["Start Time"]
        assert (durations > 0).all()

    def test_segment_confidence_in_unit_range(self, segments_df: pd.DataFrame) -> None:
        assert segments_df["Confidence"].min() > 0.0
        assert segments_df["Confidence"].max() <= 1.0


class TestDistributionDrift:
    """Calibrated against publicly reported Spotify aggregates. If a
    regenerator change tilts the dataset away from those aggregates by
    more than the tolerances below, this test catches it."""

    def test_mean_energy_in_expected_band(self, tracks_df: pd.DataFrame) -> None:
        mean = tracks_df["energy"].mean()
        # Spotify Charts aggregate mean energy ~0.65; tolerate ±0.10
        assert 0.55 <= mean <= 0.75, f"mean energy drift: {mean:.3f}"

    def test_mean_danceability_in_expected_band(self, tracks_df: pd.DataFrame) -> None:
        mean = tracks_df["danceability"].mean()
        # Spotify aggregate ~0.60; tolerate ±0.10
        assert 0.50 <= mean <= 0.70, f"mean danceability drift: {mean:.3f}"

    def test_mean_valence_in_expected_band(self, tracks_df: pd.DataFrame) -> None:
        mean = tracks_df["valence"].mean()
        # Wider tolerance — valence varies by genre mix
        assert 0.40 <= mean <= 0.60, f"mean valence drift: {mean:.3f}"

    def test_track_ids_unique(self, tracks_df: pd.DataFrame) -> None:
        assert tracks_df["track_id"].nunique() == len(tracks_df)


class TestLabelDistribution:
    """If a regenerator change collapses video segments into a single
    category, the multimodal pipeline can't surface diverse emotion
    targets. This test guards against that."""

    def test_at_least_six_label_categories_emit(self, segments_df: pd.DataFrame) -> None:
        n_categories = segments_df["Category Description"].nunique()
        assert n_categories >= 6, f"only {n_categories} categories in segment dataset"

    def test_no_single_category_dominates(self, segments_df: pd.DataFrame) -> None:
        """No category should be > 60% of segments — otherwise the
        derived emotion targets collapse."""
        cat_share = segments_df["Category Description"].value_counts(normalize=True)
        top_share = cat_share.iloc[0]
        assert top_share <= 0.60, f"category '{cat_share.index[0]}' dominates at {top_share:.0%}"
