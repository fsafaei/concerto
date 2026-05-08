# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
WORKDIR /workspace

FROM base AS dev
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --all-extras
COPY . .

FROM base AS prod
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY src ./src
ENTRYPOINT ["uv", "run", "concerto"]
