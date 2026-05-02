# Evaluation Results

This document records the measured behavior of the Music Emotion
Classifier and the end-to-end recommendation pipeline. Every number
here was produced by running `scripts/evaluate.py` and `scripts/benchmark.py`
on the committed synthetic benchmark dataset; the raw outputs are
checked in at `results/evaluation.json` and `results/benchmark.json`,
and the trained artifacts (`emotion_classifier_model.h5`, `emotion_scaler.pkl`)
ship with the repo so a reader can reproduce these numbers in one
command:

```bash
python3 scripts/evaluate.py
python3 scripts/benchmark.py
```

## Dataset

The benchmark dataset is **synthetic** by necessity. Spotify deprecated
the public `audio-features` endpoint for new third-party apps on
2024-11-27, so a fresh clone of this repo cannot legitimately fetch
10K real audio features. Rather than fabricate values inline at runtime
(the previous failure mode of this codebase), we generate a clearly-
labeled synthetic dataset whose feature distributions match publicly
reported Spotify aggregate statistics.

| Dataset | Rows | Source |
| --- | ---: | --- |
| `data/synthetic_tracks.csv` | 10,500 | `scripts/generate_synthetic_dataset.py` (12 genre profiles, distribution-matched) |
| `data/synthetic_video_segments.csv` | 1,021 | Same generator, 50 unique videos × ~22 segments each |

Every row carries `_synthetic=True` to prevent confusion with real
catalog data. Feature distributions are calibrated to within-genre
statistics from the Million Song Dataset projections and Spotify's
own audio-features documentation.

## Classifier metrics

Trained with `scripts/evaluate.py --epochs 25 --seed 42` on an 80/20
stratified train/test split (8,400 train / 2,100 test).

| Metric | Value |
| --- | ---: |
| Test accuracy | **97.4%** |
| Validation accuracy (final epoch) | 97.5% |
| Validation loss (final epoch) | 0.093 |
| Mean top-1 confidence on test | 0.94 |
| Fraction of test predictions ≥0.9 confidence | 81% |

### Per-class metrics (on 2,100-track test set)

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| happy | 0.933 | 0.959 | 0.946 | 146 |
| sad | 0.941 | 0.941 | 0.941 | 135 |
| energetic | 0.981 | 0.985 | 0.983 | 1,153 |
| calm | 0.971 | 0.966 | 0.968 | 381 |
| aggressive | 0.986 | 0.961 | 0.973 | 285 |

### Confusion matrix

See `results/confusion_matrix.png`. Diagonal dominance is strong; the
largest off-diagonal cells are `happy → energetic` and `calm → energetic`,
which is consistent with the boundary regions of the rule-based
bootstrap labeling (energy ≈ 0.7 territory).

### What the accuracy actually measures

The training labels come from a deterministic rule-based bootstrap in
`AutoLabel._assign_initial_emotions` (energy/valence/loudness/
acousticness thresholds). The classifier learns to approximate that
rule surface. **97.4% accuracy here means the MLP has generalized the
rule boundary well to held-out feature points, not that the rules are
themselves the correct ground-truth emotion taxonomy.** The path to a
stronger evaluation is replacing the rule-based labels with human-
annotated labels on a real Spotify subset — that is documented as
future work in `docs/ITERATION_LOG.md`.

## Pipeline performance benchmarks

Measured with `scripts/benchmark.py` on the same 10,500-track
synthetic dataset on a local machine (Apple Silicon, TF 2.21.0, no
GPU). Each measurement is the median of 3 timed runs after 2 warmup
runs.

### Inference latency vs batch size

| Batch size | Per-track latency | Throughput | Wall time for 10.5K |
| ---: | ---: | ---: | ---: |
| 1 | 13,261 µs | 75 tracks/s | 139.2 s |
| 32 | 424 µs | 2,359 tracks/s | 4.5 s |
| 256 | 56 µs | 17,905 tracks/s | 0.59 s |
| 1024 | **14.6 µs** | **68,649 tracks/s** | **0.15 s** |

The 1024-batch configuration runs **910× faster per track** than the
unbatched configuration. Batch size 1024 also has the lowest p99 batch
call time (15.6 ms), so the gain comes from amortizing TF graph
overhead across more tracks per call rather than improving the per-call
latency itself.

### End-to-end pipeline (load → scale → classify → score → top-10)

| Configuration | Wall time | Speedup |
| --- | ---: | ---: |
| Naive (per-call inference, Python-loop scoring) | 2.32 s | baseline |
| Optimized (batched inference, vectorized scoring) | **22 ms** | **103×** |

The optimized path uses pre-fitted scaler reuse and `np.argpartition`
for the top-10 selection; the naive path refits the scaler per call
and scores in a Python `for` loop. Both produce the same top-10 output
on the same input.

## Reproducibility

Both scripts are deterministic for fixed seeds:

```bash
python3 scripts/generate_synthetic_dataset.py    # writes data/*.csv
python3 scripts/evaluate.py --seed 42            # writes results/evaluation.json + .png
python3 scripts/benchmark.py                     # writes results/benchmark.json
```

The seed is the only knob; results match within numerical noise.
Hardware-dependent timings (the benchmarks) will vary by ~2× across
machines but the relative speedups are stable.

## Honest limitations

1. **Synthetic features ≠ real features.** Distribution-matched is not
   the same as catalog-matched. A model trained here generalizes to
   this distribution; transfer to real Spotify features is not measured.
2. **Rule-based labels.** Held-out accuracy measures rule
   approximation, not real emotion taxonomy fit.
3. **No latency budget targeting.** 14.6 µs/track is fast in absolute
   terms but the system has no SLA against which to call it
   "production-ready." Different use cases would need different
   ceilings.
4. **Single-machine benchmark.** No GPU runs, no distributed inference,
   no real Spotify API rate limits in the loop.
