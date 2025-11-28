FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Python + system libs
RUN apt-get update && \
    apt-get install -y \
        python3.12 python3.12-venv python3.12-dev python3-pip \
        build-essential \
        libffi-dev libssl-dev \
        libsqlite3-dev libbz2-dev liblzma-dev \
        tk-dev libreadline-dev zlib1g-dev \
        libprotobuf-dev protobuf-compiler \
        libgl1 libglib2.0-0 && \
    apt-get clean

# Create venv
RUN python3.12 -m venv /opt/venv

# Auto-use venv for all commands (no need to activate manually)
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Requirements
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code
COPY . .

CMD ["python", "app.py"]