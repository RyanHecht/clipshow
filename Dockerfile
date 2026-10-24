# syntax=docker/dockerfile:1
# Two images from one file:
#   web     docker build --target web -t clipshow .
#   worker  docker build --target worker --build-arg TORCH=cpu|cuda|rocm -t clipshow-worker:cpu .

FROM python:3.13-slim AS base
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fontconfig fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CLIPSHOW_DATA=/data \
    HF_HOME=/data/cache/huggingface
WORKDIR /app

FROM base AS deps
ARG TORCH=cpu
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra ${TORCH}
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra ${TORCH}

FROM base AS web
COPY --from=deps /opt/venv /opt/venv
# Query embeddings run on the CPU: fast for one short query, and the GPU (if any) is left to workers.
ENV CLIPSHOW_DEVICE=cpu CLIPSHOW_HOST=0.0.0.0 CLIPSHOW_PORT=8000
# 8000 is the app; 8001 (with CLIPSHOW_SHARE_PORT=8001) serves share links and nothing else.
EXPOSE 8000 8001
VOLUME ["/data"]
CMD ["clipshow-web"]

FROM base AS worker
# Video drivers so ffmpeg can decode on AMD/Intel GPUs (VAAPI) when /dev/dri is passed in.
RUN apt-get update \
 && apt-get install -y --no-install-recommends mesa-va-drivers intel-media-va-driver \
 && rm -rf /var/lib/apt/lists/*
COPY --from=deps /opt/venv /opt/venv
# /data holds only downloaded models and scratch space here; the web app keeps everything that matters.
ENV CLIPSHOW_SCRATCH=/data/scratch
VOLUME ["/data"]
CMD ["clipshow-worker"]
