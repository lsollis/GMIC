# Base image with CUDA 12.2 and Ubuntu 22.04
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Install system tools and Python
RUN apt-get update && apt-get install -y \
    python3 python3-dev python3-pip git curl \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python3 -m pip install --upgrade pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies directly
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    numpy>=1.14.3 \
    scipy>=1.0.0 \
    h5py>=2.7.1 \
    imageio>=2.4.1 \
    pandas>=0.22.0 \
    tqdm>=4.19.8 \
    opencv-python>=3.4.2 \
    matplotlib>=3.0.2 \
    jupyter \
    nvflare

# Default shell
CMD ["/bin/bash"]
