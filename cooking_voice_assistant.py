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
    
    # Create the model - use gemini-1.5-flash-002 for better video support
    model = genai.GenerativeModel('gemini-1.5-flash-002')
    
    # Recipe extraction prompt
    prompt = f"""
    Analyze this YouTube cooking video and extract a complete recipe. Return ONLY a valid JSON object with this exact structure:

    {{
        "title": "Recipe name from the video",
        "description": "Brief description of the dish",
        "cuisine": "Cuisine type (e.g., Italian, Indian, American)",
        "difficulty": "Easy|Medium|Hard|Expert",
        "total_time": 1800,
        "servings": {target_servings},
        "ingredients": [
            {{"name": "ingredient name", "amount": "quantity", "unit": "unit", "notes": "optional notes"}}
        ],
        "steps": [
            {{"instruction": "Detailed cooking instruction", "estimated_time": 300}}
        ],
        "chefs_wisdom": "Chef's tips and important notes",
        "scaling_notes": "Notes about scaling this recipe",
        "original_servings": 4
    }}

    Important requirements:
    - Scale all ingredients to exactly {target_servings} servings
    - Include all ingredients mentioned in the video
    - Break down into clear, sequential cooking steps
    - Estimate realistic time for each step in seconds
    - Include cooking techniques and temperatures when mentioned
    - Extract any chef tips or important notes
    - Return ONLY the JSON object, no additional text
    - Ensure all JSON fields are properly formatted
    """
    
    try:
        print("📹 Calling Gemini with proper FileData format...")
        print(f"🔗 URL being analyzed: {youtube_url}")
        
        # Use the correct FileData format for YouTube URLs
        response = model.generate_content([
            {
                "file_data": {
                    "file_uri": youtube_url
                }
            },
            prompt
        ])
        
        print("📨 Received response from Gemini")
        print(f"📄 Raw response (first 500 chars): {response.text[:500]}...")
        
        # Extract JSON from response
        response_text = response.text.strip()
        
        # Try to find JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group()
        else:
            json_text = response_text
        
        # Parse the JSON response
        recipe_data = json.loads(json_text)
        
        # Validate required fields
        required_fields = ['title', 'ingredients', 'steps']
        for field in required_fields:
            if field not in recipe_data:
                raise ValueError(f"Missing required field: {field}")
        
        print(f"✅ Successfully extracted recipe: {recipe_data.get('title', 'Unknown')}")
        print(f"📊 Ingredients: {len(recipe_data.get('ingredients', []))}")
        print(f"📝 Steps: {len(recipe_data.get('steps', []))}")
        
        return recipe_data
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response: {e}")
        print(f"📄 Raw response: {response.text[:500]}...")
        raise Exception("Failed to parse recipe data from video analysis")
        
    except Exception as e:
        print(f"❌ Error in video analysis: {str(e)}")
        print(f"📊 Error type: {type(e).__name__}")
        raise Exception(f"Video analysis failed: {str(e)}")