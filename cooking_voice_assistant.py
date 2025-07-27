#!/usr/bin/env python3

import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class RecipeRequest(BaseModel):
    youtube_url: str
    target_servings: int = 4

class RecipeResponse(BaseModel):
    title: str
    description: Optional[str] = None
    cuisine: Optional[str] = None
    difficulty: str = "Medium"
    total_time: int = 1800  # seconds
    servings: int = 4
    ingredients: list = []
    steps: list = []
    chefs_wisdom: Optional[str] = None
    scaling_notes: Optional[str] = None
    original_servings: Optional[int] = None

@app.get("/")
def read_root():
    return {"Hello": "Railway", "status": "CookMaa Backend Running", "port": os.getenv("PORT", "8000")}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    """Generate recipe from YouTube URL using Gemini 1.5 Flash"""
    
    print(f"🎬 Received recipe request for URL: {request.youtube_url}")
    print(f"👥 Target servings: {request.target_servings}")
    
    # TODO: Implement actual Gemini video analysis
    # For now, return a mock response to test iOS integration
    
    return RecipeResponse(
        title="Mock Recipe from Railway Backend",
        description=f"A delicious recipe for {request.target_servings} people from {request.youtube_url}",
        cuisine="International",
        difficulty="Medium",
        total_time=1800,
        servings=request.target_servings,
        ingredients=[
            {"name": "Sample ingredient 1", "amount": "1", "unit": "cup", "notes": ""},
            {"name": "Sample ingredient 2", "amount": "2", "unit": "tbsp", "notes": ""}
        ],
        steps=[
            {"step_number": 1, "instruction": "This is a mock step from Railway backend", "estimated_time": 300},
            {"step_number": 2, "instruction": "Another mock step", "estimated_time": 600}
        ],
        chefs_wisdom="This is mock recipe data from the Railway backend. Replace with actual Gemini integration.",
        scaling_notes=f"Scaled to {request.target_servings} servings",
        original_servings=4
    )