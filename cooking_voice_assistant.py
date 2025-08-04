#!/usr/bin/env python3

import os
import sys
import asyncio
import logging
import requests
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import required services
try:
    # Core PipeCat
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask, PipelineParams
    from pipecat.transports.services.daily import DailyParams, DailyTransport
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    
    # Services - Groq STT/TTS + Google LLM
    from pipecat.services.groq.stt import GroqSTTService
    from pipecat.services.groq.tts import GroqTTSService
    from pipecat.services.google.llm import GoogleLLMService
    
    # Gemini for LLM
    import google.generativeai as genai
    
    SERVICES_AVAILABLE = True
    print("✅ All required services imported successfully")
    
except ImportError as e:
    SERVICES_AVAILABLE = False
    print(f"❌ Import error: {e}")
    print("❌ Make sure you have: pip install 'pipecat-ai[groq,google,daily,silero]'")

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
DAILY_API_KEY = os.getenv("DAILY_API_KEY")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)

# FastAPI App
app = FastAPI(title="CookMaa Voice Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class VoiceSessionRequest(BaseModel):
    room_url: Optional[str] = None
    token: Optional[str] = None
    recipe_context: Optional[Dict[str, Any]] = None
    step_index: int = 0

class VoiceSessionResponse(BaseModel):
    status: str
    session_id: str
    room_url: str
    token: Optional[str] = None

# Global session storage
active_sessions: Dict[str, Dict[str, Any]] = {}

async def create_daily_room():
    """Create Daily.co room"""
    if not DAILY_API_KEY:
        raise Exception("Daily.co API key not configured")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DAILY_API_KEY}"
    }
    
    room_config = {
        "privacy": "public",
        "properties": {
            "enable_screenshare": False,
            "enable_chat": False,
            "start_video_off": True,
            "start_audio_off": False,
            "max_participants": 10,
            "exp": int((datetime.now().timestamp() + 3600))
        }
    }
    
    response = requests.post(
        "https://api.daily.co/v1/rooms",
        headers=headers,
        json=room_config,
        timeout=10
    )
    
    if response.status_code == 200:
        room_data = response.json()
        room_url = room_data["url"]
        room_name = room_data["name"]
        
        # Create token
        token_config = {
            "properties": {
                "room_name": room_name,
                "is_owner": False,
                "exp": int((datetime.now().timestamp() + 3600))
            }
        }
        
        token_response = requests.post(
            "https://api.daily.co/v1/meeting-tokens",
            headers=headers,
            json=token_config,
            timeout=10
        )
        
        token = token_response.json().get("token") if token_response.status_code == 200 else None
        
        logger.info(f"✅ Daily.co room created: {room_name}")
        return room_url, token
    else:
        raise Exception(f"Failed to create room: {response.status_code}")

async def create_voice_pipeline(room_url: str, token: str, recipe_context: Dict[str, Any], session_id: str):
    """Create complete working voice pipeline - Groq STT/TTS + Gemini LLM"""
    
    logger.info(f"🔧 Creating voice pipeline for session {session_id}")
    
    # Create Daily.co transport
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Kukma",
        params=DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )
    
    # Create services
    stt = GroqSTTService(api_key=GROQ_API_KEY)
    
    # Use Gemini as LLM (as requested)
    llm = GoogleLLMService(
        api_key=GOOGLE_API_KEY,
        model="gemini-1.5-flash",
    )
    
    # Use Groq TTS (standard service)
    tts = GroqTTSService(api_key=GROQ_API_KEY)
    
    # Setup conversation context
    recipe_title = recipe_context.get('title', 'cooking')
    current_step = recipe_context.get('steps', ['No recipe loaded'])[recipe_context.get('step_index', 0)]
    
    cooking_prompt = f"""You are Kukma, a helpful cooking voice assistant.

IMPORTANT: Keep responses very brief (1-2 sentences max) for voice interaction.
Be warm, encouraging, and practical.

Current Recipe: {recipe_title}
Current Step: {current_step}

Help with cooking questions, provide tips, and offer encouragement!"""

    messages = [{"role": "system", "content": cooking_prompt}]
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    
    # Create the complete pipeline - EXACTLY like the working example
    pipeline = Pipeline([
        transport.input(),              # Transport user input
        stt,                           # Groq STT
        context_aggregator.user(),     # User responses
        llm,                           # Gemini LLM  
        tts,                           # Groq TTS
        transport.output(),            # Transport bot output - CRITICAL!
        context_aggregator.assistant(), # Assistant spoken responses
    ])
    
    # Create task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )
    
    # Setup event handlers
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info(f"👋 Client connected to session {session_id}")
        # Start conversation
        messages.append({"role": "system", "content": "Say hello and ask how you can help with cooking."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"👋 Client disconnected from session {session_id}")
        # Ensure proper cleanup
        try:
            await task.cancel()
        except:
            pass
        if session_id in active_sessions:
            del active_sessions[session_id]
    
    logger.info("✅ Voice pipeline created successfully")
    return task

async def run_pipeline(session_id: str, task):
    """Run the pipeline"""
    try:
        logger.info(f"🚀 Starting pipeline for session {session_id}")
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
        logger.info(f"🏁 Pipeline completed for session {session_id}")
    except Exception as e:
        logger.error(f"❌ Pipeline error for session {session_id}: {e}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]
            logger.info(f"🧹 Session {session_id} cleaned up")

# API Endpoints
@app.get("/")
def read_root():
    return {
        "service": "CookMaa Voice Assistant",
        "status": "running",
        "version": "2.0.0",
        "services_available": SERVICES_AVAILABLE
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": bool(GROQ_API_KEY),
            "google": bool(GOOGLE_API_KEY),
            "daily": bool(DAILY_API_KEY),
            "pipecat": SERVICES_AVAILABLE
        }
    }

@app.post("/start-voice-session")
async def start_voice_session(request: VoiceSessionRequest):
    """Start voice session with working pipeline"""
    
    if not SERVICES_AVAILABLE:
        raise HTTPException(status_code=503, detail="Required services not available")
    
    if not all([GROQ_API_KEY, GOOGLE_API_KEY, DAILY_API_KEY]):
        raise HTTPException(status_code=500, detail="Missing required API keys")
    
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        logger.info(f"🎤 Starting voice session {session_id}")
        
        # Create or use room
        room_url = request.room_url
        token = request.token
        
        if not room_url:
            room_url, token = await create_daily_room()
        
        # Create pipeline
        task = await create_voice_pipeline(
            room_url=room_url,
            token=token,
            recipe_context=request.recipe_context or {},
            session_id=session_id
        )
        
        # Store session
        active_sessions[session_id] = {
            "task": task,
            "room_url": room_url,
            "token": token,
            "recipe_context": request.recipe_context,
            "created_at": datetime.now().isoformat()
        }
        
        # Start pipeline in background
        asyncio.create_task(run_pipeline(session_id, task))
        
        logger.info(f"✅ Voice session {session_id} started successfully")
        
        return VoiceSessionResponse(
            status="started",
            session_id=session_id,
            room_url=room_url,
            token=token
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to start voice session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def get_active_sessions():
    """Get active sessions"""
    sessions = []
    for session_id, session_data in active_sessions.items():
        sessions.append({
            "session_id": session_id,
            "room_url": session_data["room_url"],
            "recipe_title": session_data.get("recipe_context", {}).get("title", "Unknown")
        })
    
    return {"active_sessions": sessions, "count": len(sessions)}

@app.delete("/sessions/{session_id}")
async def stop_voice_session(session_id: str):
    """Stop voice session"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_data = active_sessions[session_id]
        task = session_data["task"]
        await task.cancel()
        del active_sessions[session_id]
        
        logger.info(f"🛑 Stopped voice session {session_id}")
        return {"status": "stopped", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"❌ Failed to stop session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoints
@app.get("/debug/test-pipeline")
async def test_pipeline():
    """Test if pipeline components work"""
    try:
        # Test service creation
        stt = GroqSTTService(api_key=GROQ_API_KEY)
        llm = GoogleLLMService(api_key=GOOGLE_API_KEY, model="gemini-1.5-flash")
        tts = GroqTTSService(api_key=GROQ_API_KEY)
        
        return {
            "status": "success",
            "services": {
                "stt": "GroqSTTService",
                "llm": "GoogleLLMService (gemini-1.5-flash)",
                "tts": "GroqTTSService"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting CookMaa Voice Assistant on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

