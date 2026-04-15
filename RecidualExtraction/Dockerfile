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
RUN pip install --no-cache-dir --no-build-isolation --no-deps .

ENV PKG_CONFIG_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib/pkgconfig"
ENV LD_LIBRARY_PATH="/app/Compressed-Video-Reader-main/ffmpeg/lib"

WORKDIR /app

COPY docker_requirements.txt /app/
RUN pip install -r docker_requirements.txt

# here to download dataset file for the video downloader.
RUN apt-get update && apt-get install -y python3 python3-pip nodejs wget gzip
RUN wget https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz -O /app/yt_bb_detection_train.csv.gz
RUN gzip -d /app/yt_bb_detection_train.csv.gz

# 7. Test to make sure it works during the build
RUN python3 -c "import cv_reader; print('SUCCESS: cv_reader is linked and working.')"