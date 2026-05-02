#!/usr/bin/env python3
"""Train + evaluate the music emotion classifier on the synthetic dataset.

Splits the 10K-track synthetic benchmark into 80/20 train/test, trains the
Keras MLP defined in AutoLabel.MusicEmotionClassifier, and reports:

  - overall accuracy
  - per-class precision / recall / F1
  - confusion matrix
  - top-1 prediction confidence distribution

Outputs:
  results/evaluation.json   — metrics, reproducible by seed
  results/confusion_matrix.png  — heatmap visualization
  emotion_classifier_model.h5 + emotion_scaler.pkl   — trained artifacts

The labels for training come from a deterministic rule-based bootstrap
(energy/valence/loudness/acousticness thresholds in
AutoLabel._assign_initial_emotions). The classifier learns to approximate
that rule surface — accuracy here measures how well the MLP generalizes
the rule to held-out feature points, not whether the rules are themselves
correct. That distinction is documented in EVALUATION.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from AutoLabel import EMOTION_CATEGORIES, FEATURE_COLUMNS, MusicEmotionClassifier  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "synthetic_tracks.csv"
RESULTS_DIR = REPO_ROOT / "results"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default=str(DATA_PATH))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not Path(args.data).exists():
        print(f"Dataset not found: {args.data}")
        return 1

    print(f"Loading {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} tracks loaded")

    classifier = MusicEmotionClassifier()
    feats_df = df[FEATURE_COLUMNS].copy()
    label_array = classifier._assign_initial_emotions(feats_df)
    label_indices = label_array.argmax(axis=1)

    print("\nLabel distribution (rule-based bootstrap):")
    label_counts = pd.Series(
        [EMOTION_CATEGORIES[i] for i in label_indices]
    ).value_counts()
    print(label_counts.to_string())

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        feats_df.values, label_array, label_indices,
        test_size=0.2, random_state=args.seed, stratify=label_indices,
    )
    print(f"\nTrain: {len(X_train)} samples, Test: {len(X_test)} samples")

    print(f"\nTraining {args.epochs} epochs...")
    classifier.train(X_train, y_train, epochs=args.epochs, validation_split=0.1)

    pred_labels, pred_probs = classifier.predict_emotion(
        pd.DataFrame(X_test, columns=FEATURE_COLUMNS)
    )
    pred_indices = np.array([EMOTION_CATEGORIES.index(lbl) for lbl in pred_labels])
    true_labels = [EMOTION_CATEGORIES[i] for i in idx_test]

    accuracy = float(accuracy_score(idx_test, pred_indices))
    report = classification_report(
        true_labels, pred_labels, labels=EMOTION_CATEGORIES,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(idx_test, pred_indices, labels=list(range(len(EMOTION_CATEGORIES))))
    confidence_top1 = pred_probs.max(axis=1)

    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nPer-class metrics:")
    for emotion in EMOTION_CATEGORIES:
        m = report.get(emotion, {})
        print(f"  {emotion:>11}: precision={m.get('precision', 0):.3f}  "
              f"recall={m.get('recall', 0):.3f}  f1={m.get('f1-score', 0):.3f}  "
              f"support={int(m.get('support', 0))}")

    print("\nConfidence distribution:")
    print(f"  mean: {confidence_top1.mean():.3f}")
    print(f"  median: {np.median(confidence_top1):.3f}")
    print(f"  fraction >= 0.9: {(confidence_top1 >= 0.9).mean():.3f}")
    print(f"  fraction <= 0.5: {(confidence_top1 <= 0.5).mean():.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "dataset": {
            "path": str(Path(args.data).relative_to(REPO_ROOT)),
            "tracks": int(len(df)),
            "synthetic": True,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        },
        "training": {
            "epochs": args.epochs,
            "seed": args.seed,
            "model_architecture": "Dense(64) → Dropout(0.3) → Dense(32) → Dropout(0.3) → Dense(5, softmax)",
            "feature_columns": FEATURE_COLUMNS,
        },
        "metrics": {
            "accuracy": round(accuracy, 4),
            "per_class": {
                emotion: {
                    "precision": round(report.get(emotion, {}).get("precision", 0), 4),
                    "recall": round(report.get(emotion, {}).get("recall", 0), 4),
                    "f1_score": round(report.get(emotion, {}).get("f1-score", 0), 4),
                    "support": int(report.get(emotion, {}).get("support", 0)),
                }
                for emotion in EMOTION_CATEGORIES
            },
            "confusion_matrix": {
                "labels": EMOTION_CATEGORIES,
                "matrix": cm.tolist(),
            },
            "confidence_top1": {
                "mean": round(float(confidence_top1.mean()), 4),
                "median": round(float(np.median(confidence_top1)), 4),
                "fraction_high_confidence_ge_0_9": round(float((confidence_top1 >= 0.9).mean()), 4),
                "fraction_low_confidence_le_0_5": round(float((confidence_top1 <= 0.5).mean()), 4),
            },
        },
    }
    eval_path = RESULTS_DIR / "evaluation.json"
    eval_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {eval_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(EMOTION_CATEGORIES)))
        ax.set_yticks(range(len(EMOTION_CATEGORIES)))
        ax.set_xticklabels(EMOTION_CATEGORIES, rotation=30, ha="right")
        ax.set_yticklabels(EMOTION_CATEGORIES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix (accuracy {accuracy:.3f}, n={len(X_test)})")
        for i in range(len(EMOTION_CATEGORIES)):
            for j in range(len(EMOTION_CATEGORIES)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        cm_path = RESULTS_DIR / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=120)
        print(f"Wrote {cm_path}")
    except ImportError:
        print("matplotlib not available, skipping confusion matrix plot")

    classifier.save_model(
        str(REPO_ROOT / "emotion_classifier_model.h5"),
        str(REPO_ROOT / "emotion_scaler.pkl"),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
