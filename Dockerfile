# Start with a minimal Debian-based Python 3.12 image
FROM python:3.12-slim

# Set up the working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip wget && \
    rm -rf /var/lib/apt/lists/*

# Check if an NVIDIA GPU is available and install minimal CUDA
RUN if command -v nvidia-smi &> /dev/null; then \
      echo "NVIDIA GPU detected. Installing minimal CUDA components..." && \
      apt-get update && apt-get install -y --no-install-recommends wget && \
      wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
      dpkg -i cuda-keyring_1.0-1_all.deb && \
      apt-get update && apt-get install -y --no-install-recommends \
      cuda-libraries-12-2 libcudnn8; \
    else \
      echo "No NVIDIA GPU found. Proceeding with CPU-only installation."; \
    fi

# Ensure dependencies are installed
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH="/root/.local/bin:$PATH"

# Default command to start the container
CMD ["bash"]
