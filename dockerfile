# Use the official NVIDIA CUDA 11.6.1 development image as the base
FROM nvidia/cuda:11.6.1-devel-ubuntu20.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    nano \
    curl \
    build-essential \             
    pkg-config \             
    python3-dev \      
    libgomp1 \ 
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.12 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Install pip and setuptools (which replaces distutils)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3 && \
    pip install --upgrade pip setuptools

# Create a working directory inside the container
WORKDIR /app

# Set environment variable so that the virtual environment is used by default
ENV PYTHONPATH="/app/xmr4el"

# Verify installation
RUN python3 --version && pip --version

# Default command
CMD ["/bin/bash"]
