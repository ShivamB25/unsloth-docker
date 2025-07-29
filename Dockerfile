# Multi-stage build for optimization
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cuda-nvcc-12-4 \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-setuptools \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create requirements file for uv
RUN echo "torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124" > /tmp/requirements.txt && \
    echo "torchvision --index-url https://download.pytorch.org/whl/cu124" >> /tmp/requirements.txt && \
    echo "torchaudio --index-url https://download.pytorch.org/whl/cu124" >> /tmp/requirements.txt && \
    echo "bitsandbytes" >> /tmp/requirements.txt && \
    echo "accelerate" >> /tmp/requirements.txt && \
    echo "xformers==0.0.29" >> /tmp/requirements.txt && \
    echo "peft" >> /tmp/requirements.txt && \
    echo "trl" >> /tmp/requirements.txt && \
    echo "triton" >> /tmp/requirements.txt && \
    echo "cut_cross_entropy" >> /tmp/requirements.txt && \
    echo "unsloth_zoo" >> /tmp/requirements.txt && \
    echo "sentencepiece" >> /tmp/requirements.txt && \
    echo "protobuf" >> /tmp/requirements.txt && \
    echo "datasets" >> /tmp/requirements.txt && \
    echo "huggingface_hub" >> /tmp/requirements.txt && \
    echo "hf_transfer" >> /tmp/requirements.txt && \
    echo "psutil" >> /tmp/requirements.txt && \
    echo "transformers" >> /tmp/requirements.txt && \
    echo "rich" >> /tmp/requirements.txt && \
    echo "diffusers" >> /tmp/requirements.txt && \
    echo "matplotlib" >> /tmp/requirements.txt && \
    echo "scikit-learn" >> /tmp/requirements.txt && \
    echo "ipywidgets" >> /tmp/requirements.txt

# Install packages using uv (much faster than pip)
RUN uv pip install --system -r /tmp/requirements.txt

# Final runtime stage
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y \
    python-is-python3 \
    python3 \
    git \
    nano \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
COPY --from=builder /usr/local/bin /usr/local/bin

# Install uv in final stage for any future package additions
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"


# Copy finetune script and entrypoint
COPY finetune.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

WORKDIR /app

# Default command if ENABLE_JUPYTER=false
CMD ["/bin/bash"]

ENTRYPOINT ["/app/entrypoint.sh"]
