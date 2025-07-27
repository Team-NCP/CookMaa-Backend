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

@app.post("/generate-recipe")
async def generate_recipe(request: RecipeRequest):
    """Generate recipe from YouTube URL using Gemini 1.5 Flash"""
    
    print(f"🎬 Received recipe request for URL: {request.youtube_url}")
    print(f"👥 Target servings: {request.target_servings}")
    
    if not GEMINI_API_KEY:
        print("❌ No Gemini API key configured, returning mock response")
        return _get_mock_response(request)
    
    try:
        # Analyze video with Gemini 1.5 Flash
        recipe_data = await analyze_youtube_video(request.youtube_url, request.target_servings)
        
        print("✅ Successfully generated recipe using Gemini 1.5 Flash")
        return RecipeResponse(**recipe_data)
        
    except Exception as e:
        print(f"❌ Error analyzing video with Gemini: {str(e)}")
        print("🔄 Falling back to mock response for development")
        return _get_mock_response(request)

def _get_mock_response(request: RecipeRequest) -> RecipeResponse:
    """Return mock response for development/testing"""
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

async def analyze_youtube_video(youtube_url: str, target_servings: int) -> Dict[str, Any]:
    """Analyze YouTube cooking video using Gemini 1.5 Flash"""
    
    print(f"🔍 Starting Gemini video analysis for: {youtube_url}")
    
    # Create the model
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Detailed prompt for recipe extraction
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
        # Upload and analyze the video
        print("📹 Uploading video to Gemini...")
        video_file = genai.upload_file(youtube_url)
        print("⏳ Waiting for video processing...")
        
        # Wait for processing to complete
        while video_file.state.name == "PROCESSING":
            await asyncio.sleep(2)
            video_file = genai.get_file(video_file.name)
        
        if video_file.state.name == "FAILED":
            raise Exception(f"Video processing failed: {video_file.state}")
        
        print("🧠 Analyzing video content with Gemini 1.5 Flash...")
        
        # Generate recipe from video
        response = model.generate_content([video_file, prompt])
        
        print("📝 Processing Gemini response...")
        
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
        raise Exception(f"Video analysis failed: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            if 'video_file' in locals():
                genai.delete_file(video_file.name)
                print("🗑️  Cleaned up uploaded video file")
        except:
            pass