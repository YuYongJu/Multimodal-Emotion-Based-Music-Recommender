# Reproducible runtime for the music-emotion recommender.
#
# Installs deps, copies source, and runs the evaluation + benchmark
# scripts in a clean environment. No external API credentials needed —
# the synthetic benchmark dataset is committed and is what the run
# exercises. Total build + run < 5 minutes on cold cache.
#
# Build:
#   docker build -t music-emotion-rec .
# Run evaluation:
#   docker run --rm music-emotion-rec python3 scripts/evaluate.py
# Run benchmark:
#   docker run --rm music-emotion-rec python3 scripts/benchmark.py
# Run tests:
#   docker run --rm music-emotion-rec pytest -q

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# librosa needs libsndfile + ffmpeg at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -e ".[dev]"

COPY . .

# Quick sanity run on build so a broken image fails fast
RUN python3 -c "import audio_features, AutoLabel, main; print('imports OK')"

ENTRYPOINT ["python3"]
CMD ["scripts/evaluate.py"]
