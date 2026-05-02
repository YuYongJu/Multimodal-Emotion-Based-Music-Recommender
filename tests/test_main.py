"""Unit tests for main.map_video_content_to_emotions and CLI surface."""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pytest

from main import build_parser, map_video_content_to_emotions


class TestMapVideoContentToEmotions:
    def test_action_labels_surface_aggressive_target(self) -> None:
        targets = map_video_content_to_emotions(
            labels=["fight", "explosion", "chase"],
            categories=["Action"],
        )
        assert "aggressive" in targets

    def test_nature_labels_surface_calm_target(self) -> None:
        targets = map_video_content_to_emotions(
            labels=["sunset", "water", "sky"],
            categories=["Nature"],
        )
        assert "calm" in targets

    def test_party_labels_surface_happy_and_energetic(self) -> None:
        targets = map_video_content_to_emotions(
            labels=["party", "dance"],
            categories=["Entertainment"],
        )
        assert "happy" in targets or "energetic" in targets

    def test_drama_category_surfaces_sad_or_calm(self) -> None:
        targets = map_video_content_to_emotions(
            labels=["cry", "tears"],
            categories=["Drama"],
        )
        assert any(t in targets for t in ("sad", "calm"))

    def test_empty_inputs_yield_default_target(self) -> None:
        targets = map_video_content_to_emotions(labels=[], categories=[])
        # Default fallback per the function docstring is energetic+happy
        assert "energetic" in targets
        assert "happy" in targets

    def test_unknown_labels_yield_default_target(self) -> None:
        targets = map_video_content_to_emotions(
            labels=["unknownword1", "completelynovel"],
            categories=["NotAKnownCategory"],
        )
        assert "energetic" in targets
        assert "happy" in targets

    def test_mixed_signals_combine_emotions(self) -> None:
        """Action + Nature in the same video → both aggressive and calm."""
        targets = map_video_content_to_emotions(
            labels=["fight", "sunset"],
            categories=["Action", "Nature"],
        )
        assert "aggressive" in targets
        assert "calm" in targets

    def test_case_insensitive_label_matching(self) -> None:
        upper = map_video_content_to_emotions(["FIGHT"], ["ACTION"])
        lower = map_video_content_to_emotions(["fight"], ["action"])
        assert "aggressive" in upper
        assert "aggressive" in lower

    def test_returns_list(self) -> None:
        result = map_video_content_to_emotions(["dance"], ["Entertainment"])
        assert isinstance(result, list)


class TestCLIParser:
    def test_help_does_not_crash(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--help"])
        assert exc.value.code == 0

    def test_no_args_returns_namespace(self) -> None:
        parser = build_parser()
        ns = parser.parse_args([])
        assert ns.fetch_spotify is False
        assert ns.analyze_video is False
        assert ns.train_model is False
        assert ns.recommend is False
        assert ns.full_pipeline is False

    def test_full_pipeline_flag(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["--full-pipeline"])
        assert ns.full_pipeline is True

    def test_playlist_id_argument(self) -> None:
        parser = build_parser()
        ns = parser.parse_args(["--fetch-spotify", "--playlist-id", "abc123"])
        assert ns.playlist_id == "abc123"


class TestMainModuleImportsCleanly:
    def test_module_imports_without_env_vars(self, monkeypatch) -> None:
        """The lazy-init refactor means main can be imported without
        SPOTIFY_* or GOOGLE_APPLICATION_CREDENTIALS being set."""
        for var in (
            "SPOTIFY_CLIENT_ID",
            "SPOTIFY_CLIENT_SECRET",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "PLAYLIST_IDS",
            "BUCKET_NAME",
        ):
            monkeypatch.delenv(var, raising=False)
        # Force re-import to verify
        import importlib

        import main

        importlib.reload(main)
        assert callable(main.map_video_content_to_emotions)
        assert callable(main.fetch_spotify_data)
        assert callable(main.analyze_video)
