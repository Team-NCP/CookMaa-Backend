#!/usr/bin/env python3

import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    port = os.getenv("PORT", "8000")
    print(f"🌐 ROOT endpoint hit - PORT env var: {port}")
    return {"Hello": "Railway", "port": port}

@app.get("/health")
def health_check():
    port = os.getenv("PORT", "8000") 
    print(f"❤️ HEALTH endpoint hit - PORT env var: {port}")
    return {"status": "healthy"}

# Add startup event to log environment
@app.on_event("startup")
async def startup_event():
    port = os.getenv("PORT", "8000")
    print(f"🚀 FastAPI starting up...")
    print(f"🌐 PORT environment variable: {port}")
    print(f"🔗 App should be accessible on all interfaces")
    print(f"❤️ Health check endpoint: /health")