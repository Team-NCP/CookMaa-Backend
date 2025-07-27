FROM python:3-alpine

WORKDIR /app

# Copy project files
COPY requirements.minimal.txt requirements.txt
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Use Railway's recommended command
CMD ["hypercorn", "app:app", "--bind", "::"]