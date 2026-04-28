# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS runtime

ARG APP_GID=1000
ARG APP_UID=1000

ENV MPLCONFIGDIR=/tmp/matplotlib \
    MUJOCO_GL=osmesa \
    PIP_NO_CACHE_DIR=1 \
    PYOPENGL_PLATFORM=osmesa \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libegl1 \
    libgl1 \
    libgl1-mesa-dri \
    libglfw3 \
    libglib2.0-0 \
    libosmesa6 \
    libsm6 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE Makefile ./
COPY configs ./configs
COPY mujoco_continuous_control ./mujoco_continuous_control
COPY src ./src
COPY tests ./tests

RUN python -m pip install --upgrade pip \
    && python -m pip install -e ".[dev]"

RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --create-home --gid appuser --shell /bin/bash --uid "${APP_UID}" appuser \
    && mkdir -p /app/assets /app/runs /tmp/matplotlib \
    && chown -R appuser:appuser /app /tmp/matplotlib

USER appuser

CMD ["python", "-m", "mujoco_continuous_control.train", "--config", "configs/smoke_test.yaml"]
