#!/usr/bin/env python3

"""
CookMaa Voice Assistant Backend
Pipecat-powered voice pipeline for cooking guidance
"""

import os
import asyncio
import logging
from typing import Optional

from pipecat.frames.frames import Frame, AudioRawFrame, TextFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.services.groq import GroqSTTService, GroqLLMService, GroqTTSService
from pipecat.services.google import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CookingVoiceAssistant:
    """Voice assistant specialized for cooking guidance"""
    
    def __init__(self):
        self.current_recipe = None
        self.current_step_index = 0
        self.user_context = {}
        
        # Initialize services
        self.stt_service = self._create_stt_service()
        self.tts_service = self._create_tts_service()
        self.llm_service = self._create_llm_service()
        self.vad_analyzer = SileroVADAnalyzer()
        
    def _create_stt_service(self) -> GroqSTTService:
        """Create Groq Whisper STT service - budget-friendly and fast"""
        return GroqSTTService(
            api_key=os.getenv("GROQ_API_KEY"),
            model="whisper-large-v3",
            language="en"  # Can be changed to auto-detect later
        )
    
    def _create_tts_service(self) -> GroqTTSService:
        """Create Groq TTS service - ultra-fast voice synthesis"""
        return GroqTTSService(
            api_key=os.getenv("GROQ_API_KEY"),
            voice=os.getenv("GROQ_VOICE", "Celeste-PlayAI"),  # Natural female voice for cooking
            model="playai-tts"  # Standard PlayAI model
        )
    
    def _create_llm_service(self) -> GroqLLMService:
        """Create Groq LLM service for fast, budget conversations"""
        return GroqLLMService(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=1024
        )
    
    def set_recipe_context(self, recipe_data: dict, step_index: int = 0):
        """Set current recipe context for voice interactions"""
        self.current_recipe = recipe_data
        self.current_step_index = step_index
        logger.info(f"🎯 Recipe context set: {recipe_data.get('title', 'Unknown')}, step {step_index + 1}")
    
    def build_system_prompt(self) -> str:
        """Build context-aware system prompt for Gemini"""
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
    
    async def handle_wake_word(self, audio_frame: AudioRawFrame) -> Optional[str]:
        """Process wake word detection and voice commands"""
        try:
            # STT processing
            text_result = await self.stt_service.process_frame(audio_frame)
            if not isinstance(text_result, TextFrame):
                return None
            
            transcript = text_result.text.lower()
            logger.info(f"🎤 Heard: {transcript}")
            
            # Wake word detection
            if "hey kukma" not in transcript and "hey cookma" not in transcript:
                return None
                
            logger.info("👂 Wake word detected!")
            
            # Process voice command
            response = await self.process_voice_command(transcript)
            return response
            
        except Exception as e:
            logger.error(f"❌ Wake word processing failed: {e}")
            return None
    
    async def process_voice_command(self, transcript: str) -> str:
        """Process voice command and generate appropriate response"""
        try:
            # Update system prompt with current context
            system_prompt = self.build_system_prompt()
            
            # Add command context to the prompt
            if "next step" in transcript:
                command_context = "The user said 'next step'. Move to the next cooking step and explain it clearly."
                self.current_step_index = min(self.current_step_index + 1, len(self.current_recipe.get('steps', [])) - 1)
            elif "repeat" in transcript:
                command_context = "The user said 'repeat that'. Repeat the current cooking step with clear instructions."
            elif "question" in transcript:
                command_context = "The user has a cooking question. Be ready to help with their specific question."
            elif "how long" in transcript:
                command_context = "The user is asking about timing. Provide timing information for the current step."
            else:
                command_context = "General cooking assistance request."
            
            # Create conversation prompt
            conversation_prompt = f"""
            {system_prompt}
            
            COMMAND CONTEXT: {command_context}
            
            User said: "{transcript}"
            
            Respond naturally and helpfully as Kukma the cooking assistant:
            """
            
            # Get LLM response
            llm_frame = TextFrame(text=conversation_prompt)
            response_frame = await self.llm_service.process_frame(llm_frame)
            
            if isinstance(response_frame, TextFrame):
                response_text = response_frame.text
                logger.info(f"🤖 Kukma responds: {response_text}")
                return response_text
            
            return "I'm here to help with your cooking!"
            
        except Exception as e:
            logger.error(f"❌ Command processing failed: {e}")
            return "Sorry, I didn't catch that. Can you repeat your request?"
    
    async def speak_response(self, text: str) -> Optional[AudioRawFrame]:
        """Convert text response to speech using TTS"""
        try:
            text_frame = TextFrame(text=text)
            audio_frame = await self.tts_service.process_frame(text_frame)
            
            if isinstance(audio_frame, AudioRawFrame):
                logger.info(f"🔊 Speaking: {text[:50]}...")
                return audio_frame
                
        except Exception as e:
            logger.error(f"❌ TTS failed: {e}")
        
        return None

class CookingPipeline:
    """Main pipeline for cooking voice assistant"""
    
    def __init__(self, room_url: str, token: str):
        self.room_url = room_url
        self.token = token
        self.assistant = CookingVoiceAssistant()
        
    async def run(self):
        """Run the cooking voice pipeline"""
        try:
            # Daily transport for real-time communication
            transport = DailyTransport(
                room_url=self.room_url,
                token=self.token,
                bot_name="Kukma",
                params=DailyParams(
                    audio_in_enabled=True,
                    audio_out_enabled=True,
                    camera_out_enabled=False,
                    vad_enabled=True,
                    vad_analyzer=self.assistant.vad_analyzer
                )
            )
            
            # Create pipeline
            pipeline = Pipeline([
                transport.audio_in,
                self.assistant.stt_service,
                self.assistant.llm_service, 
                self.assistant.tts_service,
                transport.audio_out
            ])
            
            # Run pipeline
            runner = PipelineRunner(pipeline)
            
            logger.info("🚀 Starting Kukma cooking voice assistant...")
            await runner.run()
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

# API endpoints for iOS integration
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="CookMaa Voice Assistant API")

class RecipeContext(BaseModel):
    title: str
    cuisine: Optional[str] = None
    servings: int = 4
    difficulty: str = "Medium"
    total_time_formatted: str = "30 min"
    steps: list = []
    chefs_wisdom: Optional[str] = None

class VoiceSessionRequest(BaseModel):
    room_url: str
    token: str
    recipe_context: Optional[RecipeContext] = None
    step_index: int = 0

# Global assistant instance
voice_assistant = CookingVoiceAssistant()

@app.post("/start-voice-session")
async def start_voice_session(request: VoiceSessionRequest):
    """Start a voice session for cooking guidance"""
    try:
        # Set recipe context if provided
        if request.recipe_context:
            recipe_data = request.recipe_context.dict()
            voice_assistant.set_recipe_context(recipe_data, request.step_index)
        
        # Create and run pipeline
        pipeline = CookingPipeline(request.room_url, request.token)
        
        # Run in background task
        asyncio.create_task(pipeline.run())
        
        return {
            "status": "success",
            "message": "Voice session started",
            "bot_name": "Kukma"
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice session: {e}")
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
            "current_step": step_index + 1
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to update context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CookMaa Voice Assistant",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check required environment variables
    required_env_vars = ["GROQ_API_KEY", "GEMINI_API_KEY", "DAILY_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {missing_vars}")
        exit(1)
    
    # Run the API server
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)