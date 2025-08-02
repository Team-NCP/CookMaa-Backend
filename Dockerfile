FROM python:3.11-slim

WORKDIR /app

# Add debug logging throughout the build process
RUN echo "🐋 DOCKER BUILD START: Installing system dependencies..." && date

# Install system dependencies needed for audio and compilation
RUN echo "📦 Installing system dependencies..." && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    libasound2-dev \
    libportaudio2 \
    portaudio19-dev \
    libsndfile1 \
    pkg-config \
    && echo "✅ System dependencies installed successfully" \
    && rm -rf /var/lib/apt/lists/* \
    || (echo "❌ System dependencies installation failed" && exit 1)

# Upgrade pip and install basic dependencies first
RUN echo "🐍 Upgrading pip and build tools..." && \
    pip install --upgrade pip setuptools wheel \
    && echo "✅ Pip and build tools upgraded successfully" \
    || (echo "❌ Pip upgrade failed" && exit 1)

# Copy application files
COPY requirements.txt .
COPY cooking_voice_assistant.py .
RUN echo "📁 Application files copied successfully"

# Install dependencies from requirements.txt with better error handling
RUN echo "📚 Installing dependencies from requirements.txt..." && \
    pip install --no-cache-dir -r requirements.txt \
    && echo "✅ All requirements.txt dependencies installed successfully" \
    || (echo "❌ Requirements install failed, trying step-by-step install..." && \
        pip install --no-cache-dir fastapi uvicorn requests python-dotenv pydantic google-generativeai aiofiles numpy && \
        pip install --no-cache-dir groq && \
        pip install --no-cache-dir "pipecat-ai[daily,groq,google]>=0.0.77" && \
        echo "✅ Step-by-step installation completed")

RUN echo "🎯 DOCKER BUILD COMPLETE: All installations attempted" && date

# Add debug info to startup
RUN echo "📋 Creating startup debug script..." && \
    echo '#!/bin/bash' > /app/debug_startup.sh && \
    echo 'echo "🚀 CONTAINER STARTUP: $(date)"' >> /app/debug_startup.sh && \
    echo 'echo "🐍 Python version: $(python --version)"' >> /app/debug_startup.sh && \
    echo 'echo "📦 Installed packages:"' >> /app/debug_startup.sh && \
    echo 'pip list | head -20' >> /app/debug_startup.sh && \
    echo 'echo "🔍 Environment variables:"' >> /app/debug_startup.sh && \
    echo 'env | grep -E "(PORT|GEMINI|GROQ|DAILY)" || echo "No relevant env vars found"' >> /app/debug_startup.sh && \
    echo 'echo "📁 Application files:"' >> /app/debug_startup.sh && \
    echo 'ls -la /app/' >> /app/debug_startup.sh && \
    echo 'echo "🎬 Starting application on port ${PORT:-8000}..."' >> /app/debug_startup.sh && \
    echo 'echo "🏥 Health check will be available at: http://0.0.0.0:${PORT:-8000}/health"' >> /app/debug_startup.sh && \
    echo 'python cooking_voice_assistant.py' >> /app/debug_startup.sh && \
    chmod +x /app/debug_startup.sh

# Use debug startup script
CMD ["/app/debug_startup.sh"]