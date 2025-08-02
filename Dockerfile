FROM python:3.11-slim

WORKDIR /app

# Add debug logging throughout the build process
RUN echo "🐋 DOCKER BUILD START: Installing system dependencies..." && date

# Install minimal system dependencies
RUN echo "📦 Installing basic build tools..." && \
    apt-get update && apt-get install -y \
    gcc \
    g++ \
    && echo "✅ Basic build tools installed successfully" \
    && rm -rf /var/lib/apt/lists/* \
    || (echo "❌ Basic build tools installation failed" && exit 1)

# Upgrade pip and install basic dependencies first
RUN echo "🐍 Upgrading pip and build tools..." && \
    pip install --upgrade pip setuptools wheel \
    && echo "✅ Pip and build tools upgraded successfully" \
    || (echo "❌ Pip upgrade failed" && exit 1)

# Copy application files
COPY requirements.txt .
COPY cooking_voice_assistant.py .
RUN echo "📁 Application files copied successfully"

# Install core dependencies that are guaranteed to work
RUN echo "📚 Installing CORE dependencies (must succeed)..." && \
    pip install --no-cache-dir fastapi uvicorn[standard] requests python-dotenv pydantic google-generativeai aiofiles \
    && echo "✅ CORE dependencies installed successfully" \
    || (echo "❌ CORE dependencies failed - this is critical!" && exit 1)

# Try to install optional dependencies (allow failures)
RUN echo "🔧 Installing OPTIONAL dependency: groq..." && \
    (pip install --no-cache-dir groq && echo "✅ Groq installed successfully") \
    || echo "⚠️  Groq install failed, continuing..."

RUN echo "📞 Installing OPTIONAL dependency: daily-python..." && \
    (pip install --no-cache-dir daily-python && echo "✅ Daily-python installed successfully") \
    || echo "⚠️  Daily-python install failed, continuing..."

# Install audio dependencies for Pipecat (allow failures)
RUN echo "🎵 Installing OPTIONAL audio dependencies..." && \
    (apt-get update && apt-get install -y \
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        libsndfile1 \
        pkg-config \
    && echo "✅ Audio dependencies installed successfully" \
    && rm -rf /var/lib/apt/lists/*) \
    || echo "⚠️  Audio dependencies install failed, continuing..."

RUN echo "🔢 Installing OPTIONAL dependency: numpy..." && \
    (pip install --no-cache-dir numpy && echo "✅ Numpy installed successfully") \
    || echo "⚠️  Numpy install failed, continuing..."

RUN echo "🎤 Installing OPTIONAL dependency: pipecat-ai with Daily.co integration..." && \
    (pip install --no-cache-dir "pipecat-ai[daily]" && echo "✅ Pipecat-ai with Daily.co installed successfully") \
    || echo "⚠️  Pipecat-ai install failed, continuing..."

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