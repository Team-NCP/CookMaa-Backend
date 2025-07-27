#!/usr/bin/env python3

"""
Minimal test server to debug Railway deployment
"""

import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello Railway!", "port": os.getenv("PORT", "8000")}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"Starting on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)