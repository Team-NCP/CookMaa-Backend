FROM python:3.11-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic dependencies first
RUN pip install --upgrade pip setuptools wheel

# Copy application files
COPY requirements.txt .
COPY cooking_voice_assistant.py .

# Install core dependencies that are guaranteed to work
RUN pip install --no-cache-dir fastapi uvicorn[standard] requests python-dotenv pydantic google-generativeai aiofiles

# Try to install optional dependencies (allow failures)
RUN pip install --no-cache-dir groq || echo "Groq install failed, continuing..."
RUN pip install --no-cache-dir daily-python || echo "Daily-python install failed, continuing..."

# Install audio dependencies for Pipecat (allow failures)
RUN apt-get update && apt-get install -y \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    pkg-config \
    || echo "Audio dependencies install failed, continuing..." \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy || echo "Numpy install failed, continuing..."
RUN pip install --no-cache-dir pipecat-ai || echo "Pipecat install failed, continuing..."

# Use uvicorn with IPv4 binding (Railway health check compatible)
CMD ["sh", "-c", "uvicorn cooking_voice_assistant:app --host 0.0.0.0 --port ${PORT:-8000}"]