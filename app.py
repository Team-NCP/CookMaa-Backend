#!/usr/bin/env python3

import os
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Railway", "port": os.getenv("PORT", "8000")}

@app.get("/health")
def health_check():
    return {"status": "healthy"}