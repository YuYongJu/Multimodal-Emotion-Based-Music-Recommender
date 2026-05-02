"""Runtime smoke tests — the things that don't break code logic but
break deployments: missing entrypoints, broken imports, env-var
crashes, file path issues."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestModuleImports:
    """Each module must import cleanly with no env vars set.

    The original code crashed at module import on missing
    GOOGLE_APPLICATION_CREDENTIALS. After the lazy-init refactor every
    module should import successfully and only fail at the point a
    function actually needs the missing credential.
    """

    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        for var in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
                    "GOOGLE_APPLICATION_CREDENTIALS", "PLAYLIST_IDS",
                    "BUCKET_NAME"):
            monkeypatch.delenv(var, raising=False)

    def test_audio_features_imports_cleanly(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import audio_features"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_autolabel_imports_cleanly(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import AutoLabel"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_main_imports_cleanly(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import main"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
        )
        assert result.returncode == 0, result.stderr

    def test_gvi_imports_cleanly_without_creds(self) -> None:
        """The lazy _ensure_clients refactor: import is fine, only the
        function call fails."""
        result = subprocess.run(
            [sys.executable, "-c", "import GoogleVideoIntelligenceAPI"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
        )
        assert result.returncode == 0, result.stderr


class TestCLIEntryPoints:
    def test_main_help_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
            timeout=30,
        )
        assert result.returncode == 0
        assert "fetch-spotify" in result.stdout
        assert "analyze-video" in result.stdout
        assert "train-model" in result.stdout
        assert "recommend" in result.stdout

    def test_main_no_args_prints_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "main.py"],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_clean_env(),
            timeout=30,
        )
        assert result.returncode == 0
        # No-args path prints the help message
        assert "usage:" in (result.stdout + result.stderr).lower()


class TestShippedArtifactsPresent:
    """The repo claims certain artifacts are committed (model weights,
    benchmark results, evaluation results, synthetic dataset). These
    smoke tests guard against accidental removal."""

    def test_synthetic_tracks_csv_present(self) -> None:
        path = REPO_ROOT / "data" / "synthetic_tracks.csv"
        assert path.exists(), "shipped synthetic tracks dataset is missing"
        # Quick sanity — header has the canonical feature columns
        first_line = path.read_text().splitlines()[0]
        for col in ("danceability", "energy", "valence", "tempo"):
            assert col in first_line

    def test_synthetic_video_segments_csv_present(self) -> None:
        path = REPO_ROOT / "data" / "synthetic_video_segments.csv"
        assert path.exists(), "shipped video segments dataset is missing"

    def test_trained_model_weights_present(self) -> None:
        h5 = REPO_ROOT / "emotion_classifier_model.h5"
        scaler = REPO_ROOT / "emotion_scaler.pkl"
        assert h5.exists(), "shipped model weights are missing"
        assert scaler.exists(), "shipped scaler is missing"

    def test_evaluation_results_present(self) -> None:
        path = REPO_ROOT / "results" / "evaluation.json"
        assert path.exists(), "shipped evaluation results are missing"
        import json
        data = json.loads(path.read_text())
        assert "metrics" in data
        assert "accuracy" in data["metrics"]

    def test_benchmark_results_present(self) -> None:
        path = REPO_ROOT / "results" / "benchmark.json"
        assert path.exists(), "shipped benchmark results are missing"
        import json
        data = json.loads(path.read_text())
        for key in ("inference_latency", "headline_latency", "video_pipeline",
                    "end_to_end"):
            assert key in data, f"benchmark results missing '{key}' section"

    def test_documentation_present(self) -> None:
        for doc in ("EVALUATION.md", "EDGE_CASES.md", "ITERATION_LOG.md"):
            assert (REPO_ROOT / "docs" / doc).exists(), f"docs/{doc} missing"


class TestNoCredentialFiles:
    """Defense in depth — even if .gitignore changes, this test catches
    a credential file accidentally landing in the working tree."""

    def test_no_service_account_json_in_root(self) -> None:
        suspicious = list(REPO_ROOT.glob("*service-account*.json"))
        suspicious += list(REPO_ROOT.glob("turing-terminus-*.json"))
        assert suspicious == [], f"credential-shaped JSON in repo root: {suspicious}"

    def test_no_env_file_committed(self) -> None:
        assert not (REPO_ROOT / ".env").exists(), ".env should never be committed"


def _clean_env() -> dict:
    """Subprocess env without Spotify/GCP creds, but with PATH so python runs."""
    return {k: v for k, v in os.environ.items()
            if k not in ("SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET",
                         "GOOGLE_APPLICATION_CREDENTIALS", "PLAYLIST_IDS",
                         "BUCKET_NAME")}
