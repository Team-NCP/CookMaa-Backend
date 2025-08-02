#!/usr/bin/env python3

import os
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Service imports  
from groq import Groq

# Try to import Pipecat, but continue if it fails
try:
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.services.daily import DailyParams, DailyTransport
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.frames.frames import TextFrame, EndFrame
    from pipecat.vad.silero import SileroVADAnalyzer
    PIPECAT_AVAILABLE = True
    logger.info("✅ Pipecat imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  Pipecat not available: {e}")
    PIPECAT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="CookMaa Voice Assistant", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure APIs
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DAILY_API_KEY = os.getenv("DAILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    logger.warning("⚠️  GROQ_API_KEY not found - STT/TTS will be limited")
else:
    groq_client = Groq(api_key=GROQ_API_KEY)
    logger.info("✅ Groq API configured successfully (STT/TTS)")

if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not found - LLM conversation will not work")
else:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully (LLM)")

if not DAILY_API_KEY:
    logger.warning("⚠️  DAILY_API_KEY not found - voice sessions will not work")
else:
    logger.info("✅ Daily.co API configured successfully")

# Data models
class VoiceSessionRequest(BaseModel):
    room_url: str
    token: str
    recipe_context: Optional[Dict[str, Any]] = None
    step_index: int = 0

class VoiceSessionResponse(BaseModel):
    status: str
    session_id: str
    room_url: str

class RecipeContextUpdate(BaseModel):
    title: str
    steps: List[str]
    step_index: int = 0
    chefs_wisdom: Optional[str] = None

# Global session storage
active_sessions: Dict[str, Dict[str, Any]] = {}

# Pipecat Voice Pipeline Classes
class CookingVoiceAssistant:
    """Pipecat-based cooking voice assistant"""
    
    def __init__(self, recipe_context: Optional[Dict[str, Any]] = None):
        self.recipe_context = recipe_context or {}
        self.conversation_history = []
        
    def build_cooking_prompt(self) -> str:
        """Build context-aware cooking system prompt"""
        base_prompt = """You are Kukma, a helpful cooking voice assistant. You help users cook by:

1. Answering questions about the current recipe step
2. Providing cooking tips and techniques  
3. Explaining ingredients and measurements
4. Helping with timing and temperature
5. Offering encouragement and motivation

IMPORTANT: Keep responses very brief (1-2 sentences) for voice interaction.
Be warm, encouraging, and practical. Always relate to cooking context."""

        if self.recipe_context and 'steps' in self.recipe_context:
            current_step_index = self.recipe_context.get('step_index', 0)
            steps = self.recipe_context.get('steps', [])
            
            if current_step_index < len(steps):
                current_step = steps[current_step_index]
                context_prompt = f"""

Current Recipe: {self.recipe_context.get('title', 'Unknown Recipe')}
Current Step ({current_step_index + 1}): {current_step}

When users ask questions, relate answers to this current step when relevant."""
                
                return base_prompt + context_prompt
        
        return base_prompt
    
    async def process_user_message(self, user_text: str) -> str:
        """Process user message with Gemini 1.5 Flash LLM"""
        try:
            system_prompt = self.build_cooking_prompt()
            
            # Build conversation for Gemini
            conversation_text = f"{system_prompt}\n\nUser: {user_text}"
            
            # Add recent conversation history for context
            if self.conversation_history:
                history_context = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}" 
                    for msg in self.conversation_history[-4:]  # Last 4 exchanges
                ])
                conversation_text = f"{system_prompt}\n\nRecent conversation:\n{history_context}\n\nUser: {user_text}"
            
            # Use Gemini 1.5 Flash for conversation
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(
                conversation_text,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    top_k=40,
                    top_p=0.9,
                    max_output_tokens=150  # Keep responses concise for voice
                )
            )
            
            if not response or not response.text:
                logger.warning("Empty response from Gemini")
                return "I'm sorry, I didn't catch that. Could you try again?"
            
            response_text = response.text.strip()
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response_text}
            ])
            
            # Keep only recent history to avoid token limits (Gemini free tier optimization)
            if len(self.conversation_history) > 8:
                self.conversation_history = self.conversation_history[-8:]
            
            logger.info(f"💭 Gemini response: {response_text}")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ Gemini LLM processing error: {str(e)}")
            return "I'm having trouble thinking right now. Could you repeat that?"

async def create_pipecat_pipeline(room_url: str, token: str, recipe_context: Dict[str, Any]) -> PipelineTask:
    """Create Pipecat pipeline for voice interaction"""
    
    # Create cooking assistant
    assistant = CookingVoiceAssistant(recipe_context)
    
    # Daily.co transport configuration
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Kukma",
        params=DailyParams(
            audio_out_enabled=True,
            audio_in_enabled=True,
            video_out_enabled=False,
            transcription_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer()
        )
    )
    
    # Create processors for the pipeline
    llm_context = OpenAILLMContext()
    
    # Custom processor for cooking responses
    class CookingProcessor:
        def __init__(self, assistant: CookingVoiceAssistant):
            self.assistant = assistant
            
        async def process_frame(self, frame, direction):
            if isinstance(frame, TextFrame):
                # Process user speech
                user_text = frame.text
                logger.info(f"🎤 User said: {user_text}")
                
                # Generate response
                response = await self.assistant.process_user_message(user_text)
                logger.info(f"💭 Assistant response: {response}")
                
                # Return response frame
                return TextFrame(response)
            
            return frame
    
    cooking_processor = CookingProcessor(assistant)
    
    # Create pipeline
    pipeline = Pipeline([
        transport.input(),
        cooking_processor,
        transport.output()
    ])
    
    # Create and return pipeline task
    task = PipelineTask(pipeline)
    return task

# API Endpoints
@app.get("/")
def read_root():
    features = ["gemini"]
    if PIPECAT_AVAILABLE:
        features.extend(["pipecat", "daily.co"])
    if GROQ_API_KEY:
        features.append("groq")
    
    return {
        "service": "CookMaa Voice Assistant",
        "status": "running", 
        "version": "2.0.0",
        "features": features,
        "pipecat_available": PIPECAT_AVAILABLE,
        "port": os.getenv("PORT", "8000")
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/start-voice-session")
async def start_voice_session(request: VoiceSessionRequest):
    """Start a new voice session with Daily.co + Pipecat"""
    
    if not PIPECAT_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="Pipecat voice pipeline not available. Service running in limited mode."
        )
    
    if not DAILY_API_KEY or not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="Voice session requires DAILY_API_KEY and GEMINI_API_KEY"
        )
    
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        logger.info(f"🎤 Starting voice session {session_id}")
        logger.info(f"🏠 Room URL: {request.room_url}")
        
        # Create Pipecat pipeline
        task = await create_pipecat_pipeline(
            room_url=request.room_url,
            token=request.token,
            recipe_context=request.recipe_context or {}
        )
        
        # Store session
        active_sessions[session_id] = {
            "task": task,
            "room_url": request.room_url,
            "recipe_context": request.recipe_context,
            "step_index": request.step_index
        }
        
        # Start the pipeline in background
        asyncio.create_task(run_pipeline(session_id, task))
        
        logger.info(f"✅ Voice session {session_id} started successfully")
        
        return VoiceSessionResponse(
            status="started",
            session_id=session_id,
            room_url=request.room_url
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start voice session: {str(e)}")

async def run_pipeline(session_id: str, task: PipelineTask):
    """Run Pipecat pipeline for a session"""
    try:
        logger.info(f"🚀 Running pipeline for session {session_id}")
        
        # Create pipeline runner
        runner = PipelineRunner()
        
        # Run the task
        await runner.run(task)
        
        logger.info(f"🏁 Pipeline finished for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline error for session {session_id}: {str(e)}")
    finally:
        # Clean up session
        if session_id in active_sessions:
            del active_sessions[session_id]
            logger.info(f"🧹 Cleaned up session {session_id}")

@app.post("/update-recipe-context/{session_id}")
async def update_recipe_context(session_id: str, context: RecipeContextUpdate):
    """Update recipe context for an active session"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Update stored context
        active_sessions[session_id]["recipe_context"] = {
            "title": context.title,
            "steps": context.steps,
            "step_index": context.step_index,
            "chefs_wisdom": context.chefs_wisdom
        }
        active_sessions[session_id]["step_index"] = context.step_index
        
        logger.info(f"📖 Updated recipe context for session {session_id}: {context.title}")
        
        return {"status": "updated", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to update context for session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update context: {str(e)}")

@app.get("/sessions")
async def get_active_sessions():
    """Get list of active voice sessions"""
    sessions = []
    for session_id, session_data in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "room_url": session_data["room_url"],
            "recipe_title": session_data.get("recipe_context", {}).get("title", "Unknown"),
            "step_index": session_data.get("step_index", 0)
        })
    
    return {"active_sessions": sessions, "count": len(sessions)}

@app.delete("/sessions/{session_id}")
async def stop_voice_session(session_id: str):
    """Stop a voice session"""
    
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Get task and stop it
        session_data = active_sessions[session_id]
        task = session_data["task"]
        
        # Stop the pipeline
        await task.stop()
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"🛑 Stopped voice session {session_id}")
        
        return {"status": "stopped", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to stop session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop session: {str(e)}")

@app.get("/debug/env")
def debug_environment():
    """Debug endpoint to check environment configuration"""
    return {
        "groq_api_key_configured": bool(GROQ_API_KEY),
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "daily_api_key_configured": bool(DAILY_API_KEY),
        "port": os.getenv("PORT", "not_set"),
        "environment_vars_count": len([k for k in os.environ.keys() if not k.startswith("_")])
    }

@app.get("/debug/groq")
async def test_groq():
    """Test Groq API connectivity (STT/TTS only)"""
    if not GROQ_API_KEY:
        return {"error": "No Groq API key configured"}
    
    try:
        # Test basic Groq functionality (we'll use this for STT/TTS)
        return {
            "status": "configured",
            "note": "Groq configured for STT/TTS only",
            "api_key_length": len(GROQ_API_KEY)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/debug/gemini")
async def test_gemini():
    """Test Gemini API connectivity (LLM)"""
    if not GEMINI_API_KEY:
        return {"error": "No Gemini API key configured"}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            "Say hello as a cooking assistant in one brief sentence.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=50
            )
        )
        
        return {
            "status": "success",
            "model": "gemini-1.5-flash",
            "response": response.text,
            "usage": "LLM for conversation"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

# Cost-Optimized Architecture for Free Tier:
# - Recipe Generation: iOS app → Gemini API directly (avoids server costs)
# - Voice Conversation: Railway backend → Gemini 1.5 Flash (free tier friendly)
# - STT/TTS: Groq (ultra-fast, cost-effective)
# - Total monthly cost: ~$5 Railway + ~$3 Groq = $8/month

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting CookMaa backend on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)