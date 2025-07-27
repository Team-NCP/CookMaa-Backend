FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for curl (health check)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port
EXPOSE 8000

# Health check (disabled for Railway deployment troubleshooting)
# HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#     CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["sh", "-c", "uvicorn simple_voice_assistant:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]