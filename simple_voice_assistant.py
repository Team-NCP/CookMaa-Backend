#!/usr/bin/env python3

"""
CookMaa Voice Assistant Backend - Simplified Version
Direct API integration without Pipecat framework
"""

import os
import asyncio
import logging
import json
import requests
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CookingVoiceAssistant:
    """Simplified voice assistant using direct API calls"""
    
    def __init__(self):
        self.current_recipe = None
        self.current_step_index = 0
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.daily_api_key = os.getenv("DAILY_API_KEY")
        
        # Log missing keys but don't fail initialization
        missing_keys = []
        if not self.groq_api_key:
            missing_keys.append("GROQ_API_KEY")
        if not self.gemini_api_key:
            missing_keys.append("GEMINI_API_KEY")
        if not self.daily_api_key:
            missing_keys.append("DAILY_API_KEY")
            
        if missing_keys:
            logger.warning(f"Missing API keys: {missing_keys}. Some features may not work.")
        else:
            logger.info("All API keys configured successfully")
    
    def set_recipe_context(self, recipe_data: dict, step_index: int = 0):
        """Set current recipe context for voice interactions"""
        self.current_recipe = recipe_data
        self.current_step_index = step_index
        logger.info(f"🎯 Recipe context set: {recipe_data.get('title', 'Unknown')}, step {step_index + 1}")
    
    def build_system_prompt(self) -> str:
        """Build context-aware system prompt"""
        base_prompt = """You are Kukma, an expert cooking assistant. You speak naturally and conversationally, 
        like a friendly chef teaching in their kitchen. Keep responses concise but encouraging.
        
        Your voice commands:
        - "Hey Kukma, next step" - Move to next cooking step
        - "Hey Kukma, repeat that" - Repeat current instruction
        - "Hey Kukma, I have a question" - Answer cooking questions
        - "Hey Kukma, how long should this take?" - Provide timing guidance
        
        Always be helpful, encouraging, and practical in your responses."""
        
        if not self.current_recipe:
            return base_prompt
        
        recipe = self.current_recipe
        current_step = recipe.get('steps', [])[self.current_step_index] if self.current_step_index < len(recipe.get('steps', [])) else None
        
        context_prompt = f"""
        
        CURRENT RECIPE CONTEXT:
        Recipe: {recipe.get('title', 'Unknown Recipe')}
        Cuisine: {recipe.get('cuisine', 'Unknown')}
        Servings: {recipe.get('servings', 'Unknown')}
        Difficulty: {recipe.get('difficulty', 'Unknown')}
        Total Time: {recipe.get('total_time_formatted', 'Unknown')}
        
        Current Step ({self.current_step_index + 1} of {len(recipe.get('steps', []))}):
        {current_step.get('instruction', 'No current step') if current_step else 'Recipe completed'}
        
        Chef's Wisdom: {recipe.get('chefs_wisdom', 'No additional notes')}
        
        Respond based on this recipe context. Be specific about the current step when relevant.
        """
        
        return base_prompt + context_prompt
    
    async def process_voice_command(self, transcript: str) -> str:
        """Process voice command and generate response using Groq LLM"""
        try:
            system_prompt = self.build_system_prompt()
            
            # Add command context
            if "next step" in transcript.lower():
                command_context = "The user said 'next step'. Move to the next cooking step and explain it clearly."
                self.current_step_index = min(self.current_step_index + 1, len(self.current_recipe.get('steps', [])) - 1)
            elif "repeat" in transcript.lower():
                command_context = "The user said 'repeat that'. Repeat the current cooking step with clear instructions."
            elif "question" in transcript.lower():
                command_context = "The user has a cooking question. Be ready to help with their specific question."
            elif "how long" in transcript.lower():
                command_context = "The user is asking about timing. Provide timing information for the current step."
            else:
                command_context = "General cooking assistance request."
            
            # Call Groq LLM
            response = await self._call_groq_llm(system_prompt, transcript, command_context)
            return response
            
        except Exception as e:
            logger.error(f"❌ Command processing failed: {e}")
            return "Sorry, I didn't catch that. Can you repeat your request?"
    
    async def _call_groq_llm(self, system_prompt: str, user_input: str, command_context: str) -> str:
        """Call Groq LLM API"""
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            conversation_prompt = f"""
            {system_prompt}
            
            COMMAND CONTEXT: {command_context}
            
            User said: "{user_input}"
            
            Respond naturally and helpfully as Kukma the cooking assistant:
            """
            
            data = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "system", "content": conversation_prompt}
                ],
                "max_tokens": 150,
                "temperature": 0.7
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Groq LLM failed: {response.status_code}")
                return "I'm having trouble thinking right now. Can you try again?"
                
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "Sorry, I'm having technical difficulties."
    
    async def synthesize_speech(self, text: str) -> Optional[bytes]:
        """Convert text to speech using Groq TTS"""
        try:
            url = "https://api.groq.com/openai/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "playai-tts",
                "input": text,
                "voice": os.getenv("GROQ_VOICE", "Celeste-PlayAI")
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                return response.content
            else:
                logger.error(f"TTS failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    async def create_daily_room(self) -> Optional[Dict[str, str]]:
        """Create a Daily.co room for voice session"""
        try:
            url = "https://api.daily.co/v1/rooms"
            headers = {
                "Authorization": f"Bearer {self.daily_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "properties": {
                    "max_participants": 2,
                    "enable_chat": False,
                    "enable_screenshare": False,
                    "exp": int(asyncio.get_event_loop().time()) + 3600  # 1 hour expiry
                }
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "room_url": result["url"],
                    "room_name": result["name"]
                }
            else:
                logger.error(f"Daily room creation failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Daily room error: {e}")
            return None

# FastAPI app
app = FastAPI(title="CookMaa Voice Assistant API", version="1.0.0")

# Global assistant instance
voice_assistant = CookingVoiceAssistant()

@app.on_event("startup")
async def startup_event():
    """Initialize the app on startup"""
    logger.info("🚀 CookMaa Voice Assistant starting up...")
    logger.info("✅ Voice assistant initialized successfully")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CookMaa Voice Assistant",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "test_voice": "/test-voice",
            "start_session": "/start-voice-session",
            "process_command": "/process-voice-command"
        }
    }

# Pydantic models
class RecipeContext(BaseModel):
    title: str
    cuisine: Optional[str] = None
    servings: int = 4
    difficulty: str = "Medium"
    total_time_formatted: str = "30 min"
    steps: list = []
    chefs_wisdom: Optional[str] = None

class VoiceSessionRequest(BaseModel):
    recipe_context: Optional[RecipeContext] = None
    step_index: int = 0

class VoiceCommandRequest(BaseModel):
    transcript: str
    recipe_context: Optional[RecipeContext] = None
    step_index: int = 0

@app.get("/health")
async def health_check():
    """Health check endpoint - simplified for Railway"""
    return {"status": "healthy"}

@app.post("/start-voice-session")
async def start_voice_session(request: VoiceSessionRequest):
    """Start a voice session for cooking guidance"""
    try:
        # Set recipe context if provided
        if request.recipe_context:
            recipe_data = request.recipe_context.dict()
            voice_assistant.set_recipe_context(recipe_data, request.step_index)
        
        # Create Daily room
        room_info = await voice_assistant.create_daily_room()
        if not room_info:
            raise HTTPException(status_code=500, detail="Failed to create voice room")
        
        return {
            "status": "success",
            "message": "Voice session started",
            "room_url": room_info["room_url"],
            "room_name": room_info["room_name"],
            "bot_name": "Kukma"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-voice-command")
async def process_voice_command(request: VoiceCommandRequest):
    """Process a voice command and return text + audio response"""
    try:
        # Update recipe context if provided
        if request.recipe_context:
            recipe_data = request.recipe_context.dict()
            voice_assistant.set_recipe_context(recipe_data, request.step_index)
        
        # Process the command
        response_text = await voice_assistant.process_voice_command(request.transcript)
        
        # Generate audio (optional)
        audio_data = await voice_assistant.synthesize_speech(response_text)
        
        return {
            "status": "success",
            "transcript": request.transcript,
            "response_text": response_text,
            "has_audio": audio_data is not None,
            "audio_length": len(audio_data) if audio_data else 0,
            "current_step": voice_assistant.current_step_index
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to process voice command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-recipe-context")
async def update_recipe_context(recipe_context: RecipeContext, step_index: int = 0):
    """Update the current recipe context"""
    try:
        recipe_data = recipe_context.dict()
        voice_assistant.set_recipe_context(recipe_data, step_index)
        
        return {
            "status": "success",
            "message": "Recipe context updated",
            "current_step": step_index + 1,
            "total_steps": len(recipe_data.get("steps", []))
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-voice")
async def test_voice():
    """Test voice synthesis"""
    try:
        test_text = "Hello! I'm Kukma, your cooking assistant. Let's start cooking something delicious!"
        audio_data = await voice_assistant.synthesize_speech(test_text)
        
        return {
            "status": "success" if audio_data else "failed",
            "text": test_text,
            "audio_generated": audio_data is not None,
            "audio_size": len(audio_data) if audio_data else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Voice test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Get port from environment - Railway sets this
    port = int(os.getenv("PORT", 8000))
    print(f"🌐 Railway PORT environment variable: {os.getenv('PORT', 'NOT SET')}")
    print(f"🌐 Starting server on 0.0.0.0:{port}")
    
    # Check environment variables but don't exit on missing ones
    required_env_vars = ["GROQ_API_KEY", "GEMINI_API_KEY", "DAILY_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"⚠️ Missing environment variables: {missing_vars}")
        print("Service will start in degraded mode. Add environment variables for full functionality.")
    else:
        print("✅ All environment variables configured")
    
    print("🚀 Starting CookMaa Voice Assistant API...")
    
    # Run the API server
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        log_level="info",
        access_log=True
    )