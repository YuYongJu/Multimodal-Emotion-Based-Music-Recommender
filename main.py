#!/usr/bin/env python3
"""Multimodal video-to-music recommender — CLI entry point.

Pipeline: Spotify playlist → real audio features → emotion classifier →
Google Video Intelligence labels → emotion match → ranked recommendations.

All clients are initialised lazily so the script doesn't crash at import
when an environment variable is missing for a subcommand the user isn't
running.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from dotenv import load_dotenv

from AutoLabel import FEATURE_COLUMNS, MusicEmotionClassifier

load_dotenv()

DEFAULT_SPOTIFY_FILE = "spotify_metadata.xlsx"
DEFAULT_VIDEO_FILE = "GoogleVideoIntelligenceLabelAnalyzer_results.xlsx"
DEFAULT_MODEL = "emotion_classifier_model.h5"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Required environment variable {name} is not set")
    return value


def fetch_spotify_data(
    playlist_id: str | None = None, output_file: str = DEFAULT_SPOTIFY_FILE
) -> str:
    """Fetch Spotify playlist + real audio features, write to xlsx."""
    print("Fetching Spotify metadata with real audio features...")
    from recommend_spotify_playlist_music_for_tiktok_edits import fetch_spotify_metadata

    target = playlist_id or os.getenv("PLAYLIST_IDS", "37i9dQZF1DXcBWIGoYBM5M").split(",")[0]
    df = fetch_spotify_metadata(target)
    if df.empty:
        raise SystemExit(f"Could not fetch any tracks from playlist {target}")
    df.to_excel(output_file, index=False)
    print(f"Wrote {len(df)} tracks → {output_file}")
    return output_file


def analyze_video(bucket_name: str) -> str:
    """Run Google Video Intelligence on every .mp4 in the bucket."""
    print(f"Analyzing videos in bucket {bucket_name} via Google Video Intelligence...")
    from GoogleVideoIntelligenceAPI import analyze_videos_in_bucket

    output_file = analyze_videos_in_bucket(bucket_name)
    if output_file is None:
        raise SystemExit(f"No videos found in bucket {bucket_name}")
    return output_file


def train_emotion_classifier(
    spotify_data_path: str = DEFAULT_SPOTIFY_FILE, epochs: int = 30
) -> MusicEmotionClassifier:
    print("Training emotion classifier on real audio features...")
    classifier = MusicEmotionClassifier()
    X, y = classifier.preprocess_data(spotify_data_path)
    classifier.train(X, y, epochs=epochs)
    classifier.save_model()
    return classifier


def map_video_content_to_emotions(labels: list[str], categories: list[str]) -> list[str]:
    """Map Video Intelligence labels/categories to target music emotions."""
    mapping = {
        "dance": ["energetic", "happy"],
        "performance": ["energetic"],
        "music": ["happy", "energetic"],
        "fun": ["happy"],
        "smile": ["happy"],
        "nature": ["calm"],
        "water": ["calm"],
        "sky": ["calm"],
        "fight": ["aggressive"],
        "explosion": ["aggressive"],
        "romance": ["calm", "sad"],
        "love": ["happy", "calm"],
        "food": ["happy"],
        "sports": ["energetic"],
        "game": ["energetic"],
        "cry": ["sad"],
        "tears": ["sad"],
        "night": ["calm", "sad"],
        "sunset": ["calm"],
        "party": ["happy", "energetic"],
        "entertainment": ["happy", "energetic"],
        "art": ["calm"],
        "action": ["energetic", "aggressive"],
        "drama": ["sad", "calm"],
        "comedy": ["happy"],
        "adventure": ["energetic"],
    }
    target = set()
    for term in [t.lower() for t in labels + categories]:
        for key, emotions in mapping.items():
            if key in term:
                target.update(emotions)
    if not target:
        target = {"energetic", "happy"}
    return list(target)


def recommend_music_for_video(
    video_data_path: str = DEFAULT_VIDEO_FILE,
    spotify_data_path: str = DEFAULT_SPOTIFY_FILE,
    model_path: str = DEFAULT_MODEL,
    top_n: int = 10,
) -> pd.DataFrame:
    print("Recommending music for video...")

    if not os.path.exists(video_data_path):
        raise SystemExit(
            f"{video_data_path} not found — run --analyze-video first or "
            "provide an existing video analysis xlsx"
        )
    if not os.path.exists(spotify_data_path):
        raise SystemExit(f"{spotify_data_path} not found — run --fetch-spotify first")

    video_df = pd.read_excel(video_data_path)
    label_counts = video_df["Label Description"].value_counts()
    category_counts = video_df["Category Description"].value_counts()
    top_labels = label_counts.head(5).index.tolist()
    top_categories = category_counts.head(5).index.tolist()
    print(f"Top labels:     {', '.join(top_labels)}")
    print(f"Top categories: {', '.join(top_categories)}")

    target_emotions = map_video_content_to_emotions(top_labels, top_categories)
    print(f"Target emotions: {', '.join(target_emotions)}")

    music_df = pd.read_excel(spotify_data_path)
    missing = [c for c in FEATURE_COLUMNS if c not in music_df.columns]
    if missing:
        raise SystemExit(
            f"Spotify metadata is missing audio feature columns {missing}. "
            "Regenerate with --fetch-spotify."
        )
    music_df = music_df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    classifier = MusicEmotionClassifier()
    if os.path.exists(model_path):
        try:
            classifier.load_model()
        except Exception as exc:
            print(f"Could not load model ({exc}); retraining...")
            classifier = train_emotion_classifier(spotify_data_path)
    else:
        classifier = train_emotion_classifier(spotify_data_path)

    features_df = music_df[FEATURE_COLUMNS]
    predicted_emotions, _ = classifier.predict_emotion(features_df)

    avg_video_intensity = float(video_df["Confidence"].mean())

    recommendations = pd.DataFrame(
        {
            "track_name": music_df["track_name"],
            "artist": music_df["artist"],
            "predicted_emotion": predicted_emotions,
        }
    )
    scores = []
    for i, emotion in enumerate(predicted_emotions):
        score = 0.0
        if emotion in target_emotions:
            score = 100.0
            score += min(float(music_df.iloc[i].get("popularity", 0)), 30.0)
            energy = float(music_df.iloc[i]["energy"])
            energy_match = 1.0 - abs(avg_video_intensity - energy)
            score += energy_match * 20.0
        scores.append(score)
    recommendations["match_score"] = scores

    top = recommendations.sort_values("match_score", ascending=False).head(top_n)
    print(f"\nTop {top_n} recommendations:")
    for i, (_, row) in enumerate(top.iterrows(), 1):
        print(
            f"  {i:2d}. {row['track_name']} — {row['artist']} "
            f"[{row['predicted_emotion']}, score {row['match_score']:.1f}]"
        )
    return top


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multimodal video-to-music recommender")
    parser.add_argument(
        "--fetch-spotify", action="store_true", help="Fetch Spotify metadata + audio features"
    )
    parser.add_argument("--playlist-id", type=str, help="Specific Spotify playlist ID to fetch")
    parser.add_argument(
        "--analyze-video", action="store_true", help="Run Google Video Intelligence on bucket .mp4s"
    )
    parser.add_argument(
        "--bucket-name",
        type=str,
        default=os.getenv("BUCKET_NAME", "music-emotion-classification-videos"),
    )
    parser.add_argument(
        "--train-model",
        action="store_true",
        help="Train the emotion classifier on fetched features",
    )
    parser.add_argument(
        "--recommend", action="store_true", help="Recommend music for the analyzed video"
    )
    parser.add_argument(
        "--full-pipeline", action="store_true", help="Run fetch → analyze → train → recommend"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.full_pipeline:
        args.fetch_spotify = args.analyze_video = args.train_model = args.recommend = True

    if not any([args.fetch_spotify, args.analyze_video, args.train_model, args.recommend]):
        parser.print_help()
        return 0

    spotify_path = DEFAULT_SPOTIFY_FILE
    video_path = DEFAULT_VIDEO_FILE

    if args.fetch_spotify:
        spotify_path = fetch_spotify_data(args.playlist_id)
    if args.analyze_video:
        video_path = analyze_video(args.bucket_name)
    if args.train_model:
        train_emotion_classifier(spotify_path)
    if args.recommend:
        recommend_music_for_video(video_path, spotify_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
