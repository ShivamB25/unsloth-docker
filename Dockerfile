FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Ubuntu packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    nano \
    wget \
    curl \
    vim \
    cuda-nvcc-12-4 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton && \
    pip install --no-deps cut_cross_entropy unsloth_zoo && \
    pip install sentencepiece protobuf datasets huggingface_hub hf_transfer && \
    pip install flash-attn --no-build-isolation && \
    pip install psutil transformers rich diffusers && \
    pip install matplotlib scikit-learn

# We install Unsloth at runtime in entrypoint.sh

# Copy finetune script and entrypoint
COPY finetune.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

# Default command if ENABLE_JUPYTER=false
CMD ["/bin/bash"]

ENTRYPOINT ["/app/entrypoint.sh"]
