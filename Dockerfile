FROM python:3-alpine

WORKDIR /app

# Copy project files
COPY requirements.minimal.txt requirements.txt
COPY cooking_voice_assistant.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use uvicorn with explicit IPv6 binding
CMD ["sh", "-c", "uvicorn cooking_voice_assistant:app --host :: --port ${PORT:-8000}"]