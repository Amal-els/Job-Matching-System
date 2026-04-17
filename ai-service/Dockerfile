FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and enforce stable defaults
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip setuptools wheel

# Install dependencies

# Install remaining dependencies
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]