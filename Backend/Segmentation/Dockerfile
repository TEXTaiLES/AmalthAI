FROM nvcr.io/nvidia/pytorch:22.12-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
