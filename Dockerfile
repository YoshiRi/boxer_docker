ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE}

ARG TORCH_VERSION=2.5.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        libgomp1 \
        tini \
        wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/boxer

COPY . /opt/boxer

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url "${TORCH_INDEX_URL}" "torch==${TORCH_VERSION}" \
    && python -m pip install \
        dill \
        numpy \
        opencv-python-headless \
        pillow \
        projectaria-tools \
        tqdm \
    && python -c "import cv2, dill, projectaria_tools, torch, tqdm; print('import check ok')"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
