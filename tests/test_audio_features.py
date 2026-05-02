"""Unit tests for audio_features.py — the two-source feature extractor."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import audio_features
from audio_features import (
    FEATURE_COLUMNS,
    FeatureUnavailable,
    from_preview_url,
    from_spotify,
    get_audio_features,
)


class TestFromSpotify:
    def test_returns_features_dict_when_api_succeeds(self) -> None:
        sp = MagicMock()
        sp.audio_features.return_value = [{
            "danceability": 0.7, "energy": 0.8, "key": 5, "loudness": -4.0,
            "mode": 1, "speechiness": 0.05, "acousticness": 0.1,
            "instrumentalness": 0.0, "liveness": 0.12, "valence": 0.6,
            "tempo": 120.0, "id": "abc", "type": "audio_features",
        }]
        result = from_spotify(sp, "abc")
        assert result is not None
        assert set(result.keys()) == set(FEATURE_COLUMNS)
        assert result["energy"] == 0.8
        assert result["tempo"] == 120.0

    def test_returns_none_when_api_returns_empty_list(self) -> None:
        sp = MagicMock()
        sp.audio_features.return_value = []
        assert from_spotify(sp, "abc") is None

    def test_returns_none_when_api_returns_list_with_none(self) -> None:
        """Spotify returns [None] for invalid track IDs."""
        sp = MagicMock()
        sp.audio_features.return_value = [None]
        assert from_spotify(sp, "bad") is None

    def test_returns_none_on_403_or_other_exception(self) -> None:
        """The Nov 2024 deprecation manifests as exceptions for new apps."""
        sp = MagicMock()
        sp.audio_features.side_effect = Exception("403 Forbidden")
        assert from_spotify(sp, "abc") is None

    def test_filters_to_canonical_feature_columns_only(self) -> None:
        sp = MagicMock()
        sp.audio_features.return_value = [{
            **dict.fromkeys(FEATURE_COLUMNS, 0.5),
            "uri": "spotify:track:abc",
            "track_href": "https://api.spotify.com/...",
            "analysis_url": "https://api.spotify.com/...",
        }]
        result = from_spotify(sp, "abc")
        assert result is not None
        assert "uri" not in result
        assert "track_href" not in result


class TestFromPreviewUrl:
    def test_returns_none_on_empty_preview_url(self) -> None:
        assert from_preview_url("") is None

    def test_returns_none_on_none_preview_url(self) -> None:
        assert from_preview_url(None) is None  # type: ignore[arg-type]

    def test_returns_none_when_request_fails(self, mocker) -> None:
        mocker.patch(
            "audio_features.requests.get",
            side_effect=Exception("network down"),
        )
        assert from_preview_url("https://example.com/preview.mp3") is None

    def test_returns_none_when_librosa_unavailable(self, mocker) -> None:
        """The guarded import returns None instead of raising ImportError."""
        import builtins
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "librosa":
                raise ImportError("not installed")
            return original_import(name, *args, **kwargs)

        mocker.patch("builtins.__import__", side_effect=fake_import)
        result = from_preview_url("https://example.com/preview.mp3")
        assert result is None

    @pytest.mark.slow
    def test_silent_audio_returns_feature_dict_or_none(
        self, mocker, silent_audio_buffer: bytes
    ) -> None:
        """Silent audio either yields a feature dict (with bounded values)
        or None — never raises, never NaN."""
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.content = silent_audio_buffer
        mocker.patch("audio_features.requests.get", return_value=response)
        result = from_preview_url("https://example.com/silent.wav")
        if result is not None:
            for col in FEATURE_COLUMNS:
                assert col in result
                assert result[col] is not None
            assert 50.0 <= result["tempo"] <= 200.0
            assert 0.0 <= result["energy"] <= 1.0
            assert -60.0 <= result["loudness"] <= 0.0


class TestGetAudioFeatures:
    def test_resolution_order_spotify_first(self, mocker) -> None:
        """When Spotify works, librosa path must not be touched."""
        sp = MagicMock()
        sp.audio_features.return_value = [{
            **dict.fromkeys(FEATURE_COLUMNS, 0.42), "id": "abc",
        }]
        librosa_spy = mocker.patch("audio_features.from_preview_url")
        result = get_audio_features("abc", "https://example.com/p.mp3", sp=sp)
        assert result["_source"] == "spotify_api"
        librosa_spy.assert_not_called()

    def test_falls_back_to_librosa_when_spotify_returns_none(self, mocker) -> None:
        sp = MagicMock()
        sp.audio_features.return_value = [None]
        librosa_features = dict.fromkeys(FEATURE_COLUMNS, 0.3)
        mocker.patch(
            "audio_features.from_preview_url", return_value=librosa_features
        )
        result = get_audio_features("abc", "https://example.com/p.mp3", sp=sp)
        assert result["_source"] == "librosa_preview"

    def test_raises_feature_unavailable_when_both_paths_fail(self, mocker) -> None:
        sp = MagicMock()
        sp.audio_features.side_effect = Exception("403")
        mocker.patch("audio_features.from_preview_url", return_value=None)
        with pytest.raises(FeatureUnavailable, match="abc"):
            get_audio_features("abc", "https://example.com/p.mp3", sp=sp)

    def test_no_spotify_client_uses_librosa_directly(self, mocker) -> None:
        librosa_features = dict.fromkeys(FEATURE_COLUMNS, 0.5)
        mocker.patch(
            "audio_features.from_preview_url", return_value=librosa_features
        )
        result = get_audio_features("abc", "https://example.com/p.mp3", sp=None)
        assert result["_source"] == "librosa_preview"

    def test_raises_when_no_preview_and_no_spotify(self) -> None:
        with pytest.raises(FeatureUnavailable):
            get_audio_features("abc", "", sp=None)


class TestFeatureColumnsContract:
    def test_feature_columns_match_spotify_schema(self) -> None:
        """If this fails, the classifier's input contract has drifted —
        AutoLabel.FEATURE_COLUMNS and audio_features.FEATURE_COLUMNS
        must stay aligned."""
        from AutoLabel import FEATURE_COLUMNS as classifier_cols
        assert list(audio_features.FEATURE_COLUMNS) == list(classifier_cols)

    def test_eleven_canonical_features(self) -> None:
        assert len(FEATURE_COLUMNS) == 11
