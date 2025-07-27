FROM python:3.11-slim

WORKDIR /app

# Copy minimal requirements and install
COPY requirements.minimal.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy minimal app
COPY app.py .

# Use Hypercorn with IPv6 binding for Railway
CMD ["sh", "-c", "hypercorn app:app --bind [::]:${PORT:-8000}"]