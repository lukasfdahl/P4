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

COPY . /app/

WORKDIR /app/Compressed-Video-Reader-main
RUN bash install.sh

RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install "numpy<2.0.0" pkgconfig cython
RUN pip install --no-cache-dir --no-build-isolation .

ENV PKG_CONFIG_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib"

WORKDIR /app

# 7. Test to make sure it works during the build
RUN python3 -c "import cv_reader; print('SUCCESS: cv_reader is linked and working.')"