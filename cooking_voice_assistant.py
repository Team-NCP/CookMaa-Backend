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

def parse_natural_language_recipe(response_text: str, target_servings: int) -> Dict[str, Any]:
    """Parse natural language recipe response into structured format"""
    
    # Extract sections based on headers
    sections = {}
    current_section = None
    current_content = []
    
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if line.startswith('TITLE:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'title'
            current_content = [line[6:].strip()]
        elif line.startswith('DESCRIPTION:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'description'
            current_content = [line[12:].strip()]
        elif line.startswith('INGREDIENTS:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'ingredients'
            current_content = []
        elif line.startswith('STEPS:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'steps'
            current_content = []
        elif line.startswith('CHEF_WISDOM:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'chef_wisdom'
            current_content = [line[12:].strip()]
        elif line.startswith('SCALING_NOTES:'):
            if current_section:
                sections[current_section] = '\n'.join(current_content)
            current_section = 'scaling_notes'
            current_content = [line[14:].strip()]
        else:
            if current_section:
                current_content.append(line)
    
    # Add final section
    if current_section:
        sections[current_section] = '\n'.join(current_content)
    
    # Parse ingredients
    ingredients = []
    if 'ingredients' in sections:
        for line in sections['ingredients'].split('\n'):
            line = line.strip()
            if line.startswith('- '):
                ingredient_text = line[2:].strip()
                # Simple parsing - can be enhanced
                ingredients.append({
                    "name": ingredient_text.split(',')[0] if ',' in ingredient_text else ingredient_text,
                    "amount": "1",
                    "unit": "",
                    "notes": ingredient_text
                })
    
    # Parse steps
    steps = []
    if 'steps' in sections:
        for line in sections['steps'].split('\n'):
            line = line.strip()
            if re.match(r'^\d+\.', line):
                step_text = re.sub(r'^\d+\.\s*', '', line)
                steps.append({
                    "instruction": step_text,
                    "estimated_time": 300  # Default 5 minutes
                })
    
    return {
        "title": sections.get('title', 'Unknown Recipe'),
        "description": sections.get('description', 'A delicious recipe'),
        "cuisine": "Unknown",  # Could be extracted from description
        "difficulty": "Medium",
        "total_time": len(steps) * 300,  # Estimate based on steps
        "servings": target_servings,
        "ingredients": ingredients,
        "steps": steps,
        "chefs_wisdom": sections.get('chef_wisdom', ''),
        "scaling_notes": sections.get('scaling_notes', ''),
        "original_servings": 4  # Default, could be parsed from scaling notes
    }

async def analyze_youtube_video(youtube_url: str, target_servings: int) -> Dict[str, Any]:
    """Analyze YouTube cooking video using Gemini 1.5 Flash"""
    
    print(f"🔍 Starting Gemini video analysis for: {youtube_url}")
    
    # Create the model - use gemini-1.5-flash-002 for better video support
    model = genai.GenerativeModel('gemini-1.5-flash-002')
    
    # Natural language prompt from iOS app (more reliable than JSON parsing)
    prompt = f"""
    Watch this cooking video carefully and provide a complete recipe. Act as an experienced chef teaching someone in your kitchen. Pay close attention to every detail - ingredients, techniques, timing, visual cues, and any tips or cultural context shared.

    CRITICAL: First identify how many people this recipe actually serves by watching the video context, portion sizes, and any mentions by the chef.

    INTELLIGENT SCALING APPROACH:
    - DO NOT assume the recipe serves 4 people
    - Identify actual serving size from video context (pot size, portions shown, chef's comments)
    - Use COOKING KNOWLEDGE for scaling, not just mathematical proportions
    - Apply ingredient-specific scaling rules:
      * Rice: ~1/2 cup (100g) uncooked rice per person
      * Pasta: ~100g dried pasta per person  
      * Lentils/Dal: ~1/4 cup dried lentils per person
      * Vegetables: Scale more generously (people like more veggies)
      * Spices: Scale conservatively (start with less, can add more)
      * Salt: Scale very conservatively 
      * Oil/Ghee: Scale moderately (health consideration)
    - Round to practical measurements (1/2, 1/4, 3/4 cups, not complex fractions)

    Structure your response EXACTLY as follows with these headers:

    TITLE: [Exact dish name from video]

    DESCRIPTION: [Brief description - cuisine, difficulty, time, cultural context for {target_servings} people]

    INGREDIENTS:
    - [Ingredient 1 with amount, unit, preparation notes]
    - [Ingredient 2 with amount, unit, preparation notes]
    - [Continue for all ingredients...]

    STEPS:
    1. [Very detailed step with exact measurements - like "Take 1 cup basmati rice, rinse 3 times until water runs clear, then add to your cooking pot"]
    2. [Next detailed step with specific actions and measurements]
    3. [Continue with detailed, actionable steps...]

    CHEF_WISDOM: [Rich context for voice assistant - tips, cultural notes, variations, storage]

    SCALING_NOTES: Original serves [X] people, scaled to {target_servings} using cooking knowledge

    Use simple headers (TITLE:, DESCRIPTION:, INGREDIENTS:, STEPS:, CHEF_WISDOM:, SCALING_NOTES:) for easy parsing.
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
        
        # Parse natural language response with headers
        response_text = response.text.strip()
        recipe_data = parse_natural_language_recipe(response_text, target_servings)
        
        print(f"✅ Successfully extracted recipe: {recipe_data.get('title', 'Unknown')}")
        print(f"📊 Ingredients: {len(recipe_data.get('ingredients', []))}")
        print(f"📝 Steps: {len(recipe_data.get('steps', []))}")
        
        return recipe_data
        
    except Exception as e:
        print(f"❌ Error in video analysis: {str(e)}")
        print(f"📊 Error type: {type(e).__name__}")
        raise Exception(f"Video analysis failed: {str(e)}")