FROM python:3.13-slim-bookworm

# System-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxext6 libxrender-dev \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

COPY ./requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create output directory for model saving
RUN mkdir -p /mnt/output

# Argo specific default envs
ENV PYTHONUNBUFFERED=1
ENV TRAIN=1
ENV DATASET_NUM_WORKERS=2
ENV DATASET_SURPRESS_WARNINGS=1

#! Default command: OVERRIDE IN ARGO WORKFLOW IF NEEDED
CMD [ "python", "src/main.py" ]
