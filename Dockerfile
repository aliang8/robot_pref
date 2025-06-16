# Use NVIDIA CUDA 12.4 runtime base image with Ubuntu 22.04
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    vim \
    unzip \
    libosmesa6-dev \
    libosmesa6 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglew2.2 \
    libgl1-mesa-glx \
    libglfw3 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

# Download and install MuJoCo 210
RUN mkdir -p /root/.mujoco \
    && cd /root/.mujoco \
    && wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz \
    && tar -xzf mujoco210-linux-x86_64.tar.gz \
    && rm mujoco210-linux-x86_64.tar.gz

# Set MuJoCo environment variables
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV MUJOCO_PY_MJKEY_PATH=/root/.mujoco/mjkey.txt
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin

# Create a symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Install some dependencies separately to handle potential conflicts
RUN pip install wheel setuptools

# Install Cython early as it's needed for building other packages
RUN pip install "Cython<3"

# Install PyTorch with CUDA 12.4 support first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip install git+https://github.com/Farama-Foundation/Metaworld.git@master#egg=metaworld

# Install dmc2gym
# RUN pip install git+git://github.com/denisyarats/dmc2gym.git

# Install other requirements
RUN pip install -r requirements.txt

# Run this again
RUN pip install mujoco==2.3.2

# Copy the rest of the application
COPY . .

# Set default command
CMD ["/bin/bash"] 