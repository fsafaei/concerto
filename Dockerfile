# syntax=docker/dockerfile:1.7
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0 AS base
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
ADD --checksum=sha256:cd2f11a9703e911b3766d0c9bad02837e8125f94fd045fc22d237d04445f45bf \
    https://astral.sh/uv/0.8.10/install.sh /tmp/install-uv.sh
RUN sh /tmp/install-uv.sh && rm /tmp/install-uv.sh
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
