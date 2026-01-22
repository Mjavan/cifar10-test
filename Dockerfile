FROM python:3.10-slim

WORKDIR /app

# Install minimal system deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
  && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install python deps
RUN pip install -U pip \
 && pip install -e . \
 && pip install pytest \
 && pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Default command (runs training script)
CMD ["python", "-m", "src.main_train"]








