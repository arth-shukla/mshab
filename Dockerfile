# syntax=docker/dockerfile:1
#
# ManiSkill-HAB (MS-HAB) runtime image.
#
# Follows the "Setup and Installation" section of README.md:
#   1. python >= 3.9 environment
#   2. ManiSkill3 (`mshab` branch) installed from source
#   3. `pip install -e .[train,dev]` for this repo
#   4. ycb / ReplicaCAD / ReplicaCADRearrange assets (downloaded at *run* time,
#      see `mshab-download-assets` below -- they are several GB and belong in a
#      mounted volume, not in the image)
#
# Build (from the repo root, so the project source is in the build context):
#   docker build -t mshab:latest .
#
# Run (requires the NVIDIA Container Toolkit):
#   docker run --rm -it --gpus all \
#     -v mshab-assets:/root/.maniskill \
#     -v "$PWD":/work/mshab \
#     mshab:latest
#
# Inside the container, download the assets once (they persist in the volume):
#   mshab-download-assets

ARG CUDA_IMAGE=nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
FROM ${CUDA_IMAGE}

# SAPIEN renders through Vulkan/EGL, so the container needs every driver
# capability the NVIDIA Container Toolkit can expose.
ENV NVIDIA_DRIVER_CAPABILITIES=all \
    NVIDIA_VISIBLE_DEVICES=all \
    DEBIAN_FRONTEND=noninteractive

# ---------------------------------------------------------------------------
# OS-level packages
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        bash-completion \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        git-lfs \
        htop \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libglvnd0 \
        libglx0 \
        libjpeg-dev \
        libpng-dev \
        libsm6 \
        libvulkan1 \
        libxext6 \
        libxrender1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync \
        tmux \
        unzip \
        vim \
        vulkan-tools \
        wget \
        xvfb \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Vulkan ICD + implicit layer so SAPIEN finds the NVIDIA driver inside the
# container (same files used by docker/Dockerfile).
COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

RUN ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# PyTorch (cu121, matching the CUDA base image)
# ---------------------------------------------------------------------------
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install torch torchvision torchaudio --index-url ${TORCH_INDEX_URL}

WORKDIR /work

# ---------------------------------------------------------------------------
# ManiSkill3 -- pinned to the `mshab` branch as recommended by the README
# ---------------------------------------------------------------------------
ARG MANISKILL_REPO=https://github.com/haosulab/ManiSkill.git
ARG MANISKILL_BRANCH=mshab
RUN git clone ${MANISKILL_REPO} -b ${MANISKILL_BRANCH} --single-branch /work/ManiSkill \
    && pip install -e /work/ManiSkill

# Pre-fetch the PhysX GPU binary that SAPIEN downloads on first GPU use, so the
# container works without network access at run time.
RUN python -c "exec('import sapien.physx as physx\ntry:\n  physx.enable_gpu()\nexcept Exception:\n  pass')"

# ---------------------------------------------------------------------------
# MS-HAB itself (train + dev extras)
# ---------------------------------------------------------------------------
# NOTE: `mshab` is an implicit namespace package (no top-level __init__.py), so
# the whole tree is copied before installing -- don't stub one in to win a cache
# layer, it changes how setuptools resolves the package.
COPY . /work/mshab
RUN pip install -e "/work/mshab[train,dev]"

# ---------------------------------------------------------------------------
# Assets
# ---------------------------------------------------------------------------
# MS_ASSET_DIR is where ManiSkill puts ycb / ReplicaCAD / ReplicaCADRearrange and
# where MS-HAB expects task plans, spawn data and the rearrange dataset. Mount a
# volume here to avoid re-downloading on every container.
ENV MS_ASSET_DIR=/root/.maniskill \
    MSHAB_DATASET_DIR=/root/.maniskill/data/scene_datasets/replica_cad_dataset/rearrange-dataset

# `mshab-download-assets` performs step 1 of the README's setup inside a running
# container (kept out of the image: ~several GB, and it belongs on a volume).
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    '# Downloads the assets required by MS-HAB into $MS_ASSET_DIR (default ~/.maniskill).' \
    'set -eu' \
    'for dataset in ycb ReplicaCAD ReplicaCADRearrange; do' \
    '    echo "==> downloading $dataset"' \
    '    python -m mani_skill.utils.download_asset -y "$dataset"' \
    'done' \
    'echo "assets installed under ${MS_ASSET_DIR:-$HOME/.maniskill}"' \
    > /usr/local/bin/mshab-download-assets \
    && chmod +x /usr/local/bin/mshab-download-assets

# Headless by default: offscreen rendering only, no on-screen SAPIEN viewer.
ENV SAPIEN_NO_DISPLAY=1

WORKDIR /work/mshab

CMD ["/bin/bash"]
