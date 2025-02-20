# Docker Setup and Image Build Guide

This guide provides step-by-step instructions for building and running a Docker image for the **qombating-fires** project, ensuring proper GPU access when available.

---
## Prerequisites
Before building the Docker image, ensure your system meets the following requirements:

### 1. Install Docker
- Check if Docker is installed:
  ```sh
  docker --version
  ```
- If not installed, follow the official installation guide: [Docker Install Guide](https://docs.docker.com/get-docker/)

### 2. (Optional) Enable GPU Support
If you want to use **GPU acceleration**, follow these steps:

#### 2.1 Verify NVIDIA Container Toolkit Installation
Run the following command:
```sh
dpkg -l | grep nvidia-container-toolkit
```
If the toolkit is not installed, install it using:
```sh
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
```
Then restart the Docker service:
```sh
sudo systemctl restart docker
```

#### 2.2 Configure Docker Runtime for NVIDIA (If Needed)
If GPU access does not work, configure Docker to use the NVIDIA runtime:
1. Open the Docker daemon configuration file:
   ```sh
   sudo nano /etc/docker/daemon.json
   ```
2. Add the following configuration:
   ```json
   {
     "runtimes": {
       "nvidia": {
         "path": "/usr/bin/nvidia-container-runtime",
         "runtimeArgs": []
       }
     },
     "default-runtime": "nvidia"
   }
   ```
3. Save the file and restart Docker:
   ```sh
   sudo systemctl restart docker
   ```

---
## Building the Docker Image
Once the prerequisites are met, follow these steps to build the image:

1. Navigate to the project directory:
   ```sh
   cd /path/to/qombating-fires
   ```

2. Build the Docker image:
   ```sh
   docker build -t qombat_image .
   ```

3. Verify that the image was created successfully:
   ```sh
   docker images | grep qombat_image
   ```

---
## Running the Container
### **Option 1: CPU-Only Mode**
Run the container normally if GPU access is not required:
```sh
docker run --rm -it qombat_image bash
```

### **Option 2: GPU-Enabled Mode**
If you have an NVIDIA GPU and have set up Docker for GPU access:
```sh
docker run --rm -it --gpus all qombat_image bash
```

Inside the container, test GPU access using:
```sh
python3 -c "import torch; print(torch.cuda.is_available())"
```
If the output is `True`, GPU support is working correctly.

---
## Cleaning Up Unused Docker Resources
Over time, Docker images, containers, and volumes can take up space. Use the following commands to clean up unused resources:

- Remove all stopped containers:
  ```sh
  docker container prune -f
  ```
- Remove unused images:
  ```sh
  docker image prune -f
  ```
- Remove unused volumes:
  ```sh
  docker volume prune -f
  ```
- Remove all unused data:
  ```sh
  docker system prune -a -f
  ```

---
## Summary
- Ensure **Docker** and **(optionally) NVIDIA GPU support** are properly configured.
- Build the image using `docker build -t qombat_image .`
- Run the container using either CPU (`docker run --rm -it qombat_image bash`) or GPU (`docker run --rm -it --gpus all qombat_image bash`).
- Test GPU access inside the container using PyTorch.
- Periodically clean up Docker resources to free up space.

Following these steps ensures a smooth setup for running **qombating-fires** with optimized execution on both CPU and GPU environments.

