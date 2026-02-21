# Training-oriented image sketch — pin digests in docs/REPRO.md once it builds on your cluster.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates build-essential \
    python3.10 python3.10-venv python3-pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY pyproject.toml README.md /workspace/
COPY tools /workspace/tools
COPY recipes /workspace/recipes
COPY configs /workspace/configs
COPY scripts /workspace/scripts

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
RUN pip install -U pip && pip install -e ".[data,train,eval]"

# Install PyTorch with the correct cu124 wheel index for your stack, then re-install if needed:
# pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.5.1
