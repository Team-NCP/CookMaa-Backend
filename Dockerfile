FROM python:3-alpine

WORKDIR /app

# Copy project files
COPY requirements.txt .
COPY cooking_voice_assistant.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use uvicorn with IPv4 binding (Railway health check compatible)
CMD ["sh", "-c", "uvicorn cooking_voice_assistant:app --host 0.0.0.0 --port ${PORT:-8000}"]