FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    yasm \
    pkg-config \
    cmake \
    wget \
    sudo \
    bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Compressed-Video-Reader-main /app/Compressed-Video-Reader-main

WORKDIR /app/Compressed-Video-Reader-main
RUN bash install.sh

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir "numpy<2.0.0" pkgconfig flow-vis opencv-python

# This finds "numpy>=1.17.0" in the setup files and replaces it with "numpy<2.0.0" (attempt to forcully make it install the right numpy version)
RUN sed -i 's/numpy>=1.17.0/numpy<2.0.0/g' setup.py pyproject.toml || true

RUN rm -f pyproject.toml

RUN pip install --no-cache-dir --no-build-isolation --no-deps .

ENV PKG_CONFIG_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib"

WORKDIR /app

# 7. Test to make sure it works during the build
RUN python3 -c "import cv_reader; print('SUCCESS: cv_reader is linked and working.')"