FROM docker.io/nvidia/cuda:12.1.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    gdal-bin \
    libgdal-dev \
    build-essential \
    git \
    wget \
    nano \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    nnunetv2 \
    numpy \
    scikit-image \
    tqdm \
    opencv-python \
    Pillow \
    pandas \
    matplotlib

ENV nnUNet_raw="/workspace/nnUNet_raw"
ENV nnUNet_preprocessed="/workspace/nnUNet_preprocessed"
ENV nnUNet_results="/workspace/nnUNet_results"

# Script che crea i symlink a runtime trovando automaticamente la versione driver
RUN echo '#!/bin/bash\n\
NVIDIA_LIB=$(find /usr/lib/x86_64-linux-gnu-host -name "libnvidia-ml.so.*" 2>/dev/null | head -1)\n\
if [ -n "$NVIDIA_LIB" ]; then\n\
    ln -sf "$NVIDIA_LIB" /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1\n\
fi\n\
exec "$@"' > /entrypoint.sh && chmod +x /entrypoint.sh

CMD ["sleep", "infinity"]