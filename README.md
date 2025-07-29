# Unsloth Training Environment

This repository contains an optimized Docker-based environment for fine-tuning large language models using Unsloth. Built with performance optimizations and modern tooling for efficient AI model training.

## Features

- **🚀 Fast Package Management**: Uses [uv](https://github.com/astral-sh/uv) for 50% faster Python package installations
- **🏗️ Multi-stage Build**: Optimized Docker build process for smaller final images
- **⚡ CUDA 12.4.1**: Latest CUDA version supported by Unsloth with Ubuntu 22.04 LTS
- **🔧 Development Ready**: Includes Jupyter, nano, vim, and essential development tools
- **📦 Pre-configured**: All Unsloth dependencies pre-installed and optimized

## Prerequisites

- Docker installed on your system
- NVIDIA GPU with CUDA 12.4+ driver support
- NVIDIA Container Toolkit installed

## Running the Container
The image is hosted on [Docker Hub](https://hub.docker.com/repository/docker/shivamb25/unsloth-dev)
You can run the container in different modes:

1. Run with bash shell only (default):
```bash
docker run -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  shivamb25/unsloth-dev
```

2. Run with Jupyter Notebook:
```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ENABLE_JUPYTER=true \
  shivamb25/unsloth-dev
```

3. Run with JupyterLab (enhanced interface):
```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ENABLE_JUPYTER=true \
  -e USE_JUPYTERLAB=true \
  shivamb25/unsloth-dev
```

## Jupyter Interfaces

When enabled with the `ENABLE_JUPYTER=true` option, the container starts a Jupyter server on port 8888. You can also specify `USE_JUPYTERLAB=true` to start JupyterLab instead of the traditional Notebook.

You can access either interface by navigating to:
```
http://localhost:8888
```

Jupyter provides an interactive environment for running and modifying the fine-tuning script. JupyterLab offers a more integrated development experience with a file browser, multiple tabs, and extensions.

## Hugging Face Cache

### Why Mount the Cache?

The Hugging Face cache stores downloaded models, tokenizers, and other assets locally. Mounting this cache directory has several benefits:

1. **Faster Startup**: Avoid re-downloading models on each container run
2. **Disk Space Efficiency**: Prevent duplicate model downloads
3. **Bandwidth Conservation**: Reduce unnecessary network traffic
4. **Offline Capability**: Access previously downloaded models without internet connection

### Cache Location

- Host Machine: `$HOME/.cache/huggingface/`
- Container: `/root/.cache/huggingface/`

### Additional Mount Options

For more granular control, you can mount specific cache subdirectories:

```bash
docker run -it --gpus all \
  -v $HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub \  # Model weights and files
  -v $HOME/.cache/huggingface/datasets:/root/.cache/huggingface/datasets \  # Dataset cache
  -v $HOME/.cache/huggingface/accelerate:/root/.cache/huggingface/accelerate \  # Accelerate configs
  shivamb25/unsloth-dev
```

## Container Features

- **🎯 Unsloth Optimized**: Pre-configured with PyTorch 2.6.0, CUDA 12.4, and all Unsloth dependencies
- **📈 Performance Optimized**: Multi-stage Docker build with uv package manager for faster builds
- **🔬 Jupyter Integration**: Optional Jupyter Notebook/JupyterLab server for interactive development
- **🛠️ Development Tools**: Includes nano, vim, wget, curl, git, and other utilities
- **⚡ GPU Acceleration**: Full CUDA 12.4.1 support for efficient model training on modern GPUs
- **📦 Pre-built Dependencies**: All packages pre-compiled including xformers, bitsandbytes, and transformers

## Technical Specifications

- **Base Image**: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **Python Version**: 3.10 (Ubuntu 22.04 default)
- **PyTorch**: 2.6.0 with CUDA 12.4 support
- **Package Manager**: uv (Rust-based, high-performance Python package installer)
- **Build System**: Multi-stage Docker build for optimized image size

## Advanced Configuration

### Flash Attention

Flash Attention is pre-configured in the container. If you need to reinstall or update it:

```bash
apt-get update; apt-get install -y cuda-nvcc-12-4; rm -rf /var/lib/apt/lists/*
uv pip install --system flash-attn --no-build-isolation
```

### Adding New Packages

The container includes uv package manager for fast installations:

```bash
# Install a new package
uv pip install --system package-name

# Install from requirements file
uv pip install --system -r requirements.txt
```

### Building from Source

To build the container locally:

```bash
git clone https://github.com/ShivamB25/unsloth-docker.git
cd unsloth-docker
docker build -t unsloth-dev .
```

The multi-stage build process optimizes for both build speed (using uv) and final image size.

## Performance Notes

- **Build Time**: ~50% faster package installation compared to pip thanks to uv
- **Image Size**: Optimized through multi-stage build removing build dependencies
- **Memory Efficiency**: Unsloth's 4-bit quantization reduces VRAM usage by up to 70%
- **CUDA Compatibility**: Supports CUDA compute capability 6.0+ (Pascal architecture and newer)
