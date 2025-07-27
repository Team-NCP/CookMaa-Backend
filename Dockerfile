FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a startup script
RUN echo '#!/bin/bash\n\
echo "=== Railway Deployment Debug ==="\n\
echo "PORT environment variable: ${PORT:-NOT_SET}"\n\
echo "All environment variables:"\n\
env | grep -E "(PORT|RAILWAY)" || echo "No PORT/RAILWAY vars found"\n\
echo "================================"\n\
\n\
# Use PORT from Railway or default to 8000\n\
export APP_PORT=${PORT:-8000}\n\
echo "Starting uvicorn on port: $APP_PORT"\n\
\n\
exec uvicorn simple_voice_assistant:app --host 0.0.0.0 --port $APP_PORT --log-level info\n\
' > /app/start.sh && chmod +x /app/start.sh

# Run the startup script
CMD ["/app/start.sh"]