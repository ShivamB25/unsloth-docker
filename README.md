# Unsloth Training Environment

This repository contains a Docker-based environment for fine-tuning the Mistral Small 24B Instruct model using Unsloth, any model can be trained on the environment, Mistral happens to be as an example.

## Prerequisites

- Docker installed on your system
- NVIDIA GPU with CUDA support
- NVIDIA Container Toolkit installed

## Running the Container

You can run the container in different modes:

1. Run with bash shell only (default):
```bash
docker run -it --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  barrahome/unsloth-docker
```

2. Run with Jupyter Notebook:
```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ENABLE_JUPYTER=true \
  barrahome/unsloth-docker
```

3. Run with JupyterLab (enhanced interface):
```bash
docker run -it --gpus all \
  -p 8888:8888 \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -e ENABLE_JUPYTER=true \
  -e USE_JUPYTERLAB=true \
  barrahome/unsloth-docker
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
  barrahome/unsloth-docker
```

## Container Features

- **Automatic Unsloth Installation**: The container automatically installs the latest version of Unsloth at startup
- **Jupyter Integration**: Optional Jupyter Notebook server for interactive development
- **Development Tools**: Includes nano, vim, wget, curl, and other utilities for convenience
- **GPU Acceleration**: Full CUDA support for efficient model training