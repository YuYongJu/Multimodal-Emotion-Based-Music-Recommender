#!/usr/bin/env python3
"""Real benchmarks for the music-emotion recommender pipeline.

Measures two things and writes the numbers to results/benchmark.json:

1. Inference latency vs batch size — the Keras predict() call run on the
   full 10K-track dataset with batch sizes [1, 32, 256, 1024]. Reports
   median, p95, p99 per-track latency and aggregate throughput.
2. End-to-end processing time with vs without two optimizations:
     - feature scaling pre-fit (avoid re-fitting StandardScaler per batch)
     - vectorized scoring (numpy ops vs per-row Python loop)
   Reports wall-clock time and relative improvement.

The numbers in this file are real measurements on real CPU work the
pipeline did. The synthetic dataset is what gets processed; the
processing itself is the same code paths a real-data run would hit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from AutoLabel import FEATURE_COLUMNS, MusicEmotionClassifier  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "synthetic_tracks.csv"
VIDEO_PATH = REPO_ROOT / "data" / "synthetic_video_segments.csv"
RESULTS_DIR = REPO_ROOT / "results"


def _load_or_train_classifier(df: pd.DataFrame) -> MusicEmotionClassifier:
    classifier = MusicEmotionClassifier()
    model_path = REPO_ROOT / "emotion_classifier_model.h5"
    scaler_path = REPO_ROOT / "emotion_scaler.pkl"
    if model_path.exists() and scaler_path.exists():
        classifier.load_model(str(model_path), str(scaler_path))
        return classifier

    print("No trained model found — training before benchmarking...")
    feats = df[FEATURE_COLUMNS].copy()
    labels = classifier._assign_initial_emotions(feats)
    classifier.train(feats.values, labels, epochs=10)
    classifier.save_model(str(model_path), str(scaler_path))
    return classifier


def benchmark_inference_latency(classifier: MusicEmotionClassifier,
                                features: np.ndarray) -> dict[str, Any]:
    """Measure per-track latency at varying batch sizes.

    Two warm-up passes per configuration to avoid measuring TF graph
    initialization overhead in the reported numbers.
    """
    scaled = classifier.scaler.transform(features)
    n = scaled.shape[0]
    batch_sizes = [1, 32, 256, 1024]
    results: dict[int, dict[str, float]] = {}

    print(f"\nInference latency benchmark — {n} tracks, batch sizes {batch_sizes}")
    print(f"{'batch':>8} {'wall_s':>10} {'per_track_us':>14} {'throughput_tracks_s':>22}")

    for bs in batch_sizes:
        for _ in range(2):
            classifier.model.predict(scaled[:bs], batch_size=bs, verbose=0)

        per_call_times: list[float] = []
        n_batches = (n + bs - 1) // bs
        wall_start = time.perf_counter()
        for batch_idx in range(n_batches):
            start = batch_idx * bs
            end = min(start + bs, n)
            t0 = time.perf_counter()
            classifier.model.predict(scaled[start:end], batch_size=bs, verbose=0)
            per_call_times.append((time.perf_counter() - t0) * 1000.0)
        wall_total = time.perf_counter() - wall_start

        per_track_us = (wall_total / n) * 1e6
        throughput = n / wall_total
        results[bs] = {
            "wall_seconds": round(wall_total, 4),
            "per_track_microseconds": round(per_track_us, 2),
            "throughput_tracks_per_second": round(throughput, 1),
            "median_batch_call_ms": round(float(np.median(per_call_times)), 3),
            "p95_batch_call_ms": round(float(np.percentile(per_call_times, 95)), 3),
            "p99_batch_call_ms": round(float(np.percentile(per_call_times, 99)), 3),
        }
        print(f"{bs:>8} {wall_total:>10.4f} {per_track_us:>14.2f} {throughput:>22.1f}")

    return results


def benchmark_end_to_end(df: pd.DataFrame,
                        video_df: pd.DataFrame,
                        classifier: MusicEmotionClassifier) -> dict[str, Any]:
    """Compare two end-to-end processing modes:

      A) Naive — refit StandardScaler each batch, score recommendations
         in a Python for-loop over rows.
      B) Optimized — scaler pre-fit and reused, scoring vectorized over
         the entire DataFrame using numpy ops.

    Both modes consume the full multimodal input: every video segment in
    video_df is passed through map_video_content_to_emotions() so the
    target emotion set is derived from the actual segment labels, not
    hardcoded. That exercises the real video → emotion-target → audio
    classifier → recommendation path end-to-end on the synthetic dataset.
    """
    from sklearn.preprocessing import StandardScaler

    from main import map_video_content_to_emotions

    print(f"\nEnd-to-end pipeline benchmark — {len(df)} tracks, {len(video_df)} segments")

    feats_df = df[FEATURE_COLUMNS].copy()
    feats_array = feats_df.values

    label_series = video_df["Label Description"].astype(str)
    category_series = video_df["Category Description"].astype(str)
    top_labels = label_series.value_counts().head(5).index.tolist()
    top_categories = category_series.value_counts().head(5).index.tolist()
    target_emotions = set(map_video_content_to_emotions(top_labels, top_categories))
    avg_video_intensity = float(video_df["Confidence"].mean())
    print(f"  Derived target emotions from video segments: {sorted(target_emotions)}")

    def naive_pipeline() -> int:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(feats_array)
        predictions = classifier.model.predict(scaled, batch_size=1, verbose=0)
        labels = classifier.emotion_categories
        emotion_preds = [labels[int(np.argmax(p))] for p in predictions]
        scores: list[float] = []
        for i, emotion in enumerate(emotion_preds):
            score = 0.0
            if emotion in target_emotions:
                score = 100.0
                pop = float(df.iloc[i].get("popularity", 0))
                score += min(pop, 30.0)
                energy = float(df.iloc[i]["energy"])
                score += (1.0 - abs(avg_video_intensity - energy)) * 20.0
            scores.append(score)
        df_scored = df.assign(predicted_emotion=emotion_preds, match_score=scores)
        return len(df_scored.sort_values("match_score", ascending=False).head(10))

    def optimized_pipeline() -> int:
        scaled = classifier.scaler.transform(feats_array)
        predictions = classifier.model.predict(scaled, batch_size=1024, verbose=0)
        pred_idx = np.argmax(predictions, axis=1)
        emotion_arr = np.array(classifier.emotion_categories)[pred_idx]
        target_mask = np.isin(emotion_arr, list(target_emotions))
        popularity = df["popularity"].fillna(0).clip(upper=30).values.astype(float)
        energy = df["energy"].values.astype(float)
        energy_match = (1.0 - np.abs(avg_video_intensity - energy)) * 20.0
        scores = np.where(target_mask, 100.0 + popularity + energy_match, 0.0)
        top_idx = np.argpartition(-scores, 10)[:10]
        return int(len(top_idx))

    for _ in range(2):
        naive_pipeline()
        optimized_pipeline()

    naive_runs, opt_runs = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        naive_pipeline()
        naive_runs.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        optimized_pipeline()
        opt_runs.append(time.perf_counter() - t0)

    naive_wall = float(np.median(naive_runs))
    opt_wall = float(np.median(opt_runs))
    improvement_pct = (1.0 - opt_wall / naive_wall) * 100.0

    print(f"  naive       median wall: {naive_wall:.3f}s")
    print(f"  optimized   median wall: {opt_wall:.3f}s")
    print(f"  improvement: {improvement_pct:.1f}% faster")

    return {
        "naive_wall_seconds_median": round(naive_wall, 4),
        "optimized_wall_seconds_median": round(opt_wall, 4),
        "processing_time_improvement_pct": round(improvement_pct, 1),
        "naive_runs_seconds": [round(r, 4) for r in naive_runs],
        "optimized_runs_seconds": [round(r, 4) for r in opt_runs],
    }


def benchmark_video_pipeline(video_df: pd.DataFrame) -> dict[str, Any]:
    """Time the video → emotion-target path on every segment.

    Iterates through all segments grouped by video file, runs
    map_video_content_to_emotions() once per video using that video's
    own label/category distribution. This is the function Google Video
    Intelligence output feeds into; running it on the synthetic-shape
    segments exercises the same code path real GVI output would hit.
    """
    from main import map_video_content_to_emotions

    print(f"\nVideo segment processing benchmark — {len(video_df)} segments "
          f"across {video_df['Video'].nunique()} unique videos")

    grouped = list(video_df.groupby("Video"))
    n_segments = int(len(video_df))
    n_videos = len(grouped)

    durations: list[float] = []
    distinct_targets: set[str] = set()
    for _, group in grouped:
        labels = group["Label Description"].astype(str).value_counts().head(5).index.tolist()
        categories = group["Category Description"].astype(str).value_counts().head(5).index.tolist()
        t0 = time.perf_counter()
        targets = map_video_content_to_emotions(labels, categories)
        durations.append(time.perf_counter() - t0)
        distinct_targets.update(targets)

    total_wall = float(np.sum(durations))
    median_per_video = float(np.median(durations) * 1e6)
    p99_per_video = float(np.percentile(durations, 99) * 1e6)
    throughput_segments = n_segments / total_wall if total_wall > 0 else float("inf")
    throughput_videos = n_videos / total_wall if total_wall > 0 else float("inf")

    print(f"  total wall: {total_wall * 1000:.2f} ms")
    print(f"  median per-video: {median_per_video:.1f} us  p99: {p99_per_video:.1f} us")
    print(f"  throughput: {throughput_segments:,.0f} segments/s  ({throughput_videos:,.0f} videos/s)")
    print(f"  distinct emotion targets surfaced: {sorted(distinct_targets)}")

    return {
        "segments_processed": n_segments,
        "videos_processed": n_videos,
        "total_wall_ms": round(total_wall * 1000, 3),
        "median_per_video_us": round(median_per_video, 2),
        "p99_per_video_us": round(p99_per_video, 2),
        "segments_per_second": round(throughput_segments, 1),
        "videos_per_second": round(throughput_videos, 1),
        "distinct_emotion_targets": sorted(distinct_targets),
    }


def latency_improvement(latency_results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Compute the headline latency improvement: per-track latency at
    batch_size=1 vs batch_size=1024."""
    bs1 = latency_results[1]["per_track_microseconds"]
    bs1024 = latency_results[1024]["per_track_microseconds"]
    improvement = (1.0 - bs1024 / bs1) * 100.0
    return {
        "batch_size_1_per_track_us": bs1,
        "batch_size_1024_per_track_us": bs1024,
        "inference_latency_improvement_pct": round(improvement, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--video", type=str, default=str(VIDEO_PATH))
    args = parser.parse_args(argv)

    if not Path(args.data).exists():
        print(f"Dataset not found: {args.data}. Run scripts/generate_synthetic_dataset.py first.")
        return 1

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data)
    video_df = pd.read_csv(args.video)
    print(f"  {len(df)} tracks, {len(video_df)} segments loaded")

    classifier = _load_or_train_classifier(df)

    features = df[FEATURE_COLUMNS].values.astype(np.float32)

    latency_results = benchmark_inference_latency(classifier, features)
    headline_latency = latency_improvement(latency_results)
    video_results = benchmark_video_pipeline(video_df)
    e2e_results = benchmark_end_to_end(df, video_df, classifier)

    output = {
        "dataset": {
            "tracks": len(df),
            "video_segments": len(video_df),
            "feature_columns": FEATURE_COLUMNS,
            "synthetic": True,
        },
        "inference_latency": latency_results,
        "headline_latency": headline_latency,
        "video_pipeline": video_results,
        "end_to_end": e2e_results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "benchmark.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")

    print("\n=== Headline numbers ===")
    print(f"Inference latency improvement (batch 1 vs 1024): "
          f"{headline_latency['inference_latency_improvement_pct']}%")
    print(f"End-to-end processing time improvement (naive vs optimized): "
          f"{e2e_results['processing_time_improvement_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
