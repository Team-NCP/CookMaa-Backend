#!/usr/bin/env python3

import os
import asyncio
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv
import json
import re

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in environment variables")
    print("💡 Set GEMINI_API_KEY environment variable for video analysis")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini API configured successfully")

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

@app.get("/debug/env")
def debug_environment():
    """Debug endpoint to check environment configuration"""
    return {
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_api_key_length": len(GEMINI_API_KEY) if GEMINI_API_KEY else 0,
        "port": os.getenv("PORT", "not_set"),
        "environment_vars_count": len([k for k in os.environ.keys() if not k.startswith("_")])
    }

@app.get("/debug/gemini")
async def test_gemini():
    """Test Gemini API connectivity"""
    if not GEMINI_API_KEY:
        return {"error": "No API key configured"}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Say 'Hello from Gemini!' in JSON format: {\"message\": \"your response\"}")
        return {
            "status": "success",
            "response": response.text,
            "model": "gemini-1.5-flash"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.post("/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    """Generate recipe from YouTube URL using Gemini 1.5 Flash"""
    
    print(f"🎬 Received recipe request for URL: {request.youtube_url}")
    print(f"👥 Target servings: {request.target_servings}")
    
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")
    
    try:
        # Analyze video with Gemini 1.5 Flash
        recipe_data = await analyze_youtube_video(request.youtube_url, request.target_servings)
        
        print("✅ Successfully generated recipe using Gemini 1.5 Flash")
        return RecipeResponse(**recipe_data)
        
    except Exception as e:
        print(f"❌ Error analyzing video with Gemini: {str(e)}")
        print(f"📊 Error type: {type(e).__name__}")
        if hasattr(e, '__dict__'):
            print(f"📄 Error details: {e.__dict__}")
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


async def analyze_youtube_video(youtube_url: str, target_servings: int) -> Dict[str, Any]:
    """Analyze YouTube cooking video using Gemini 1.5 Flash"""
    
    print(f"🔍 Starting Gemini video analysis for: {youtube_url}")
    
    # Create the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Simple natural language prompt to see what Gemini actually returns
    prompt = f"""
    Watch this YouTube cooking video at {youtube_url} and tell me exactly what recipe is being made. 
    
    What is the dish called? What ingredients do they use? What are the cooking steps?
    
    Please analyze the actual video content and give me a detailed description of what you see.
    """
    
    try:
        print("📹 Calling Gemini to analyze YouTube video...")
        print(f"🔗 URL being sent: {youtube_url}")
        print(f"📝 Prompt: {prompt[:100]}...")
        
        # Generate analysis from YouTube URL
        response = model.generate_content(prompt)
        
        print("📨 Received response from Gemini")
        print(f"📄 Raw response (first 500 chars): {response.text[:500]}...")
        print(f"📏 Full response length: {len(response.text)} characters")
        
        # For now, just return the raw response to see what we're getting
        return {
            "title": "DEBUG - Raw Gemini Response",
            "description": response.text,
            "cuisine": "Unknown",
            "difficulty": "Medium", 
            "total_time": 1800,
            "servings": target_servings,
            "ingredients": [{"name": "DEBUG", "amount": "1", "unit": "test", "notes": "Raw Gemini response in description field"}],
            "steps": [{"instruction": "Check the description field for the actual Gemini response", "estimated_time": 60}],
            "chefs_wisdom": f"URL tested: {youtube_url}",
            "scaling_notes": "This is a debug response",
            "original_servings": 4
        }
        
    except Exception as e:
        print(f"❌ ACTUAL Error in Gemini call: {str(e)}")
        print(f"📊 Error type: {type(e).__name__}")
        print(f"📄 Error details: {e}")
        raise Exception(f"Real Gemini error: {str(e)}")