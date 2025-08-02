FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Pipecat audio processing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY cooking_voice_assistant.py .

# Install Python dependencies with verbose output for debugging
RUN pip install --no-cache-dir --verbose -r requirements.txt

# Use uvicorn with IPv4 binding (Railway health check compatible)
CMD ["sh", "-c", "uvicorn cooking_voice_assistant:app --host 0.0.0.0 --port ${PORT:-8000}"]