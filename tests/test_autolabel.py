"""Unit tests for AutoLabel.MusicEmotionClassifier."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
import pytest

from AutoLabel import (
    EMOTION_CATEGORIES,
    FEATURE_COLUMNS,
    MissingFeaturesError,
    MusicEmotionClassifier,
)


class TestRuleBasedLabeling:
    @pytest.fixture
    def classifier(self) -> MusicEmotionClassifier:
        return MusicEmotionClassifier()

    def test_high_energy_high_valence_classified_happy(
        self, classifier: MusicEmotionClassifier
    ) -> None:
        feats = pd.DataFrame([{
            "danceability": 0.7, "energy": 0.85, "key": 5, "loudness": -3.0,
            "mode": 1, "speechiness": 0.05, "acousticness": 0.1,
            "instrumentalness": 0.0, "liveness": 0.1, "valence": 0.85,
            "tempo": 130.0,
        }])
        labels = classifier._assign_initial_emotions(feats)
        assert labels.shape == (1, len(EMOTION_CATEGORIES))
        winner = EMOTION_CATEGORIES[int(np.argmax(labels[0]))]
        assert winner == "happy"

    def test_low_energy_low_valence_classified_sad(
        self, classifier: MusicEmotionClassifier
    ) -> None:
        feats = pd.DataFrame([{
            "danceability": 0.3, "energy": 0.2, "key": 0, "loudness": -20.0,
            "mode": 0, "speechiness": 0.05, "acousticness": 0.6,
            "instrumentalness": 0.4, "liveness": 0.1, "valence": 0.15,
            "tempo": 70.0,
        }])
        labels = classifier._assign_initial_emotions(feats)
        winner = EMOTION_CATEGORIES[int(np.argmax(labels[0]))]
        assert winner == "sad"

    def test_high_energy_loud_low_valence_classified_aggressive(
        self, classifier: MusicEmotionClassifier
    ) -> None:
        feats = pd.DataFrame([{
            "danceability": 0.5, "energy": 0.9, "key": 7, "loudness": -3.0,
            "mode": 0, "speechiness": 0.05, "acousticness": 0.05,
            "instrumentalness": 0.1, "liveness": 0.1, "valence": 0.2,
            "tempo": 140.0,
        }])
        labels = classifier._assign_initial_emotions(feats)
        winner = EMOTION_CATEGORIES[int(np.argmax(labels[0]))]
        assert winner == "aggressive"

    def test_low_energy_high_acousticness_classified_calm(
        self, classifier: MusicEmotionClassifier
    ) -> None:
        feats = pd.DataFrame([{
            "danceability": 0.4, "energy": 0.3, "key": 4, "loudness": -15.0,
            "mode": 1, "speechiness": 0.05, "acousticness": 0.85,
            "instrumentalness": 0.5, "liveness": 0.1, "valence": 0.55,
            "tempo": 85.0,
        }])
        labels = classifier._assign_initial_emotions(feats)
        winner = EMOTION_CATEGORIES[int(np.argmax(labels[0]))]
        assert winner == "calm"

    def test_label_output_is_one_hot_per_row(
        self, classifier: MusicEmotionClassifier, small_track_dataframe: pd.DataFrame
    ) -> None:
        feats = small_track_dataframe[FEATURE_COLUMNS]
        labels = classifier._assign_initial_emotions(feats)
        # Each row sums to exactly 1.0 (one-hot)
        assert np.allclose(labels.sum(axis=1), 1.0)
        # Every row has exactly one 1.0 cell
        assert ((labels == 1.0).sum(axis=1) == 1).all()

    def test_labels_emit_all_categories_in_canonical_order(
        self, classifier: MusicEmotionClassifier, small_track_dataframe: pd.DataFrame
    ) -> None:
        feats = small_track_dataframe[FEATURE_COLUMNS]
        labels = classifier._assign_initial_emotions(feats)
        # Output column count matches emotion category count
        assert labels.shape[1] == len(EMOTION_CATEGORIES) == 5

    def test_deterministic_for_same_input(
        self, classifier: MusicEmotionClassifier, small_track_dataframe: pd.DataFrame
    ) -> None:
        feats = small_track_dataframe[FEATURE_COLUMNS]
        labels1 = classifier._assign_initial_emotions(feats)
        labels2 = classifier._assign_initial_emotions(feats)
        assert np.array_equal(labels1, labels2)


class TestFeaturesFromDataframe:
    def test_raises_missing_features_error_when_columns_absent(self) -> None:
        classifier = MusicEmotionClassifier()
        df = pd.DataFrame([{"track_id": "abc", "track_name": "x", "artist": "y"}])
        with pytest.raises(MissingFeaturesError, match="audio feature columns missing"):
            classifier._features_from_dataframe(df)

    def test_extracts_only_canonical_columns(
        self, small_track_dataframe: pd.DataFrame
    ) -> None:
        classifier = MusicEmotionClassifier()
        feats = classifier._features_from_dataframe(small_track_dataframe)
        assert list(feats.columns) == FEATURE_COLUMNS

    def test_preserves_row_count(self, small_track_dataframe: pd.DataFrame) -> None:
        classifier = MusicEmotionClassifier()
        feats = classifier._features_from_dataframe(small_track_dataframe)
        assert len(feats) == len(small_track_dataframe)


class TestPreprocessData:
    def test_drops_rows_with_nan_features(
        self, tmp_path, small_track_dataframe: pd.DataFrame
    ) -> None:
        df = small_track_dataframe.copy()
        df.loc[0, "energy"] = np.nan
        df.loc[5, "tempo"] = np.nan
        path = tmp_path / "small.xlsx"
        df.to_excel(path, index=False)

        classifier = MusicEmotionClassifier()
        X, y = classifier.preprocess_data(str(path))
        assert len(X) == len(df) - 2

    def test_raises_when_all_rows_have_nan(
        self, tmp_path, small_track_dataframe: pd.DataFrame
    ) -> None:
        df = small_track_dataframe.copy()
        df["energy"] = np.nan
        path = tmp_path / "all_nan.xlsx"
        df.to_excel(path, index=False)

        classifier = MusicEmotionClassifier()
        with pytest.raises(MissingFeaturesError, match="NaN"):
            classifier.preprocess_data(str(path))


class TestPredictAndSaveLoad:
    @pytest.mark.slow
    def test_predict_emotion_returns_label_per_row(
        self, small_track_dataframe: pd.DataFrame
    ) -> None:
        classifier = MusicEmotionClassifier()
        feats = small_track_dataframe[FEATURE_COLUMNS]
        labels_y = classifier._assign_initial_emotions(feats)
        classifier.train(feats.values, labels_y, epochs=2,
                        validation_split=0.2, batch_size=8)
        predicted, probs = classifier.predict_emotion(feats)
        assert len(predicted) == len(feats)
        assert all(lbl in EMOTION_CATEGORIES for lbl in predicted)
        assert probs.shape == (len(feats), len(EMOTION_CATEGORIES))
        # Softmax outputs sum to ~1 per row
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    @pytest.mark.slow
    def test_save_load_round_trip_preserves_predictions(
        self, tmp_path, small_track_dataframe: pd.DataFrame
    ) -> None:
        classifier = MusicEmotionClassifier()
        feats = small_track_dataframe[FEATURE_COLUMNS]
        labels_y = classifier._assign_initial_emotions(feats)
        classifier.train(feats.values, labels_y, epochs=2,
                        validation_split=0.2, batch_size=8)
        predictions_before, _ = classifier.predict_emotion(feats)

        model_path = tmp_path / "model.h5"
        scaler_path = tmp_path / "scaler.pkl"
        classifier.save_model(str(model_path), str(scaler_path))

        loaded = MusicEmotionClassifier()
        loaded.load_model(str(model_path), str(scaler_path))
        predictions_after, _ = loaded.predict_emotion(feats)

        assert predictions_before == predictions_after

    def test_predict_without_train_raises(
        self, small_track_dataframe: pd.DataFrame
    ) -> None:
        classifier = MusicEmotionClassifier()
        with pytest.raises(RuntimeError, match="Model not trained"):
            classifier.predict_emotion(small_track_dataframe[FEATURE_COLUMNS])
