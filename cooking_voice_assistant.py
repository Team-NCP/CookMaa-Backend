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

# Enable comprehensive debug logging from the start
print(f"🐍 PYTHON STARTUP: {datetime.now()}")
print(f"🐍 Python version: {sys.version}")
print(f"🐍 Working directory: {os.getcwd()}")
print(f"🐍 Python path: {sys.path[:3]}...")  # First 3 entries

# Try to import optional service dependencies with detailed logging
print("📦 IMPORTS: Starting dependency imports...")

try:
    print("📦 IMPORTS: Attempting to import Groq...")
    from groq import Groq
    GROQ_AVAILABLE = True
    print("✅ IMPORTS: Groq imported successfully")
except ImportError as e:
    GROQ_AVAILABLE = False
    Groq = None
    print(f"⚠️  IMPORTS: Groq import failed: {e}")

# Try to import Pipecat with multiple approaches for different versions
PIPECAT_AVAILABLE = False
PIPECAT_IMPORTS = {}

try:
    print("📦 IMPORTS: Attempting Pipecat v0.0.77+ imports (method 1)...")
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineTask
    from pipecat.transports.services.daily import DailyParams, DailyTransport
    from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.frames.frames import TextFrame, EndFrame
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    
    PIPECAT_IMPORTS = {
        'Pipeline': Pipeline,
        'PipelineRunner': PipelineRunner, 
        'PipelineTask': PipelineTask,
        'DailyParams': DailyParams,
        'DailyTransport': DailyTransport,
        'LLMContext': OpenAILLMContext,
        'FrameProcessor': FrameProcessor,
        'TextFrame': TextFrame,
        'EndFrame': EndFrame,
        'SileroVADAnalyzer': SileroVADAnalyzer
    }
    PIPECAT_AVAILABLE = True
    print("✅ IMPORTS: Pipecat v0.0.77+ imported successfully (method 1)")
except ImportError as e:
    print(f"⚠️  IMPORTS: Pipecat v0.0.77+ method failed: {e}")
    
    try:
        print("📦 IMPORTS: Attempting Pipecat v0.0.45 fallback imports (method 2)...")
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask
        from pipecat.transports.services.daily import DailyParams, DailyTransport
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.processors.frame_processor import FrameDirection
        from pipecat.frames.frames import TextFrame, EndFrame
        from pipecat.audio.vad.silero import SileroVADAnalyzer
        
        PIPECAT_IMPORTS = {
            'Pipeline': Pipeline,
            'PipelineRunner': PipelineRunner, 
            'PipelineTask': PipelineTask,
            'DailyParams': DailyParams,
            'DailyTransport': DailyTransport,
            'LLMContext': OpenAILLMContext,
            'FrameProcessor': None,  # Different in v0.0.45
            'TextFrame': TextFrame,
            'EndFrame': EndFrame,
            'SileroVADAnalyzer': SileroVADAnalyzer
        }
        PIPECAT_AVAILABLE = True
        print("✅ IMPORTS: Pipecat v0.0.45 imported successfully (method 2)")
    except ImportError as e2:
        print(f"⚠️  IMPORTS: Pipecat v0.0.45 method failed: {e2}")
        
        try:
            print("📦 IMPORTS: Attempting basic Pipecat discovery (method 3)...")
            import pipecat
            print(f"📦 IMPORTS: Pipecat version: {getattr(pipecat, '__version__', 'unknown')}")
            print(f"📦 IMPORTS: Pipecat location: {pipecat.__file__}")
            print(f"📦 IMPORTS: Available pipecat modules: {list(pipecat.__dict__.keys())[:10]}")
            PIPECAT_AVAILABLE = False  # Don't enable without proper imports
            print("⚠️  IMPORTS: Pipecat found but specific imports failed")
        except ImportError as e3:
            print(f"⚠️  IMPORTS: All Pipecat import methods failed")
            print(f"⚠️  IMPORTS: Method 1 error: {e}")
            print(f"⚠️  IMPORTS: Method 2 error: {e2}")
            print(f"⚠️  IMPORTS: Method 3 error: {e3}")
            PIPECAT_AVAILABLE = False

print("📦 IMPORTS: Core dependency imports completed")

# Load environment variables
print("🔧 CONFIG: Loading environment variables...")
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("🔧 CONFIG: Environment and logging setup completed")

# Log import status with detailed info
print("📊 STATUS: Checking import availability...")
if PIPECAT_AVAILABLE:
    logger.info("✅ Pipecat imported successfully")
    print("✅ STATUS: Pipecat - AVAILABLE")
else:
    logger.warning("⚠️  Pipecat not available")
    print("⚠️  STATUS: Pipecat - NOT AVAILABLE")

if GROQ_AVAILABLE:
    logger.info("✅ Groq imported successfully")
    print("✅ STATUS: Groq - AVAILABLE")
else:
    logger.warning("⚠️  Groq not available")
    print("⚠️  STATUS: Groq - NOT AVAILABLE")

print("🚀 FASTAPI: Creating FastAPI application...")
app = FastAPI(title="CookMaa Voice Assistant", version="2.0.0")

print("🌐 FASTAPI: Adding CORS middleware...")
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("✅ FASTAPI: CORS middleware added successfully")

# Add startup event for Railway healthcheck
@app.on_event("startup")
async def startup_event():
    print("🎬 FASTAPI-EVENT: Startup event triggered")
    logger.info("FastAPI application startup completed")
    print("✅ FASTAPI-EVENT: Application is ready to serve requests")

# Configure APIs
print("🔑 API-CONFIG: Reading API keys from environment...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DAILY_API_KEY = os.getenv("DAILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(f"🔑 API-CONFIG: GROQ_API_KEY = {'SET' if GROQ_API_KEY else 'NOT SET'}")
print(f"🔑 API-CONFIG: DAILY_API_KEY = {'SET' if DAILY_API_KEY else 'NOT SET'}")
print(f"🔑 API-CONFIG: GEMINI_API_KEY = {'SET' if GEMINI_API_KEY else 'NOT SET'}")

print("🔧 API-CONFIG: Configuring API clients...")

if not GROQ_API_KEY:
    logger.warning("⚠️  GROQ_API_KEY not found - STT/TTS will be limited")
    print("⚠️  API-CONFIG: Groq - NO API KEY")
    groq_client = None
elif not GROQ_AVAILABLE:
    logger.warning("⚠️  Groq library not available - STT/TTS will be limited")
    print("⚠️  API-CONFIG: Groq - LIBRARY NOT AVAILABLE")
    groq_client = None
else:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("✅ Groq API configured successfully (STT/TTS)")
        print("✅ API-CONFIG: Groq - CONFIGURED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"❌ Groq API configuration failed: {e}")
        print(f"❌ API-CONFIG: Groq - CONFIGURATION FAILED: {e}")
        groq_client = None

if not GEMINI_API_KEY:
    logger.warning("⚠️  GEMINI_API_KEY not found - LLM conversation will not work")
    print("⚠️  API-CONFIG: Gemini - NO API KEY")
else:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("✅ Gemini API configured successfully (LLM)")
        print("✅ API-CONFIG: Gemini - CONFIGURED SUCCESSFULLY")
    except Exception as e:
        logger.error(f"❌ Gemini API configuration failed: {e}")
        print(f"❌ API-CONFIG: Gemini - CONFIGURATION FAILED: {e}")

if not DAILY_API_KEY:
    logger.warning("⚠️  DAILY_API_KEY not found - voice sessions will not work")
    print("⚠️  API-CONFIG: Daily.co - NO API KEY")
else:
    logger.info("✅ Daily.co API configured successfully")
    print("✅ API-CONFIG: Daily.co - CONFIGURED SUCCESSFULLY")

print("🎯 API-CONFIG: All API configuration attempts completed")

# Daily.co Room Management
async def create_daily_room():
    """Create a new Daily.co room for voice session"""
    if not DAILY_API_KEY:
        raise Exception("Daily.co API key not configured")
    
    print("🏠 DAILY: Creating new Daily.co room...")
    logger.info("🏠 Creating new Daily.co room")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {DAILY_API_KEY}"
        }
        
        room_config = {
            "privacy": "private",
            "properties": {
                "enable_screenshare": False,
                "enable_chat": False,
                "enable_knocking": False,
                "enable_prejoin_ui": False,
                "start_video_off": True,
                "start_audio_off": False,
                "max_participants": 2,
                "exp": int((datetime.now().timestamp() + 3600))  # 1 hour expiry
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
            
            print(f"✅ DAILY: Room created successfully - {room_name}")
            print(f"🔗 DAILY: Room URL: {room_url}")
            logger.info(f"✅ Daily.co room created: {room_name} -> {room_url}")
            
            return room_url, None  # No token needed for private rooms
        else:
            error_msg = f"Failed to create room: {response.status_code} - {response.text}"
            print(f"❌ DAILY: {error_msg}")
            logger.error(f"❌ Daily.co room creation failed: {error_msg}")
            raise Exception(error_msg)
            
    except Exception as e:
        error_msg = f"Daily.co room creation error: {str(e)}"
        print(f"❌ DAILY: {error_msg}")
        logger.error(f"❌ {error_msg}")
        raise Exception(error_msg)

# Data models
class VoiceSessionRequest(BaseModel):
    room_url: Optional[str] = None  # Optional - will create if not provided
    token: Optional[str] = None     # Optional - will create if not provided
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

async def create_pipecat_pipeline(room_url: str, token: str, recipe_context: Dict[str, Any]):
    """Create Pipecat pipeline for voice interaction"""
    
    if not PIPECAT_AVAILABLE or not PIPECAT_IMPORTS:
        raise Exception("Pipecat not available - voice pipeline cannot be created")
    
    # Create cooking assistant
    assistant = CookingVoiceAssistant(recipe_context)
    
    # Get the imported classes
    Pipeline = PIPECAT_IMPORTS['Pipeline']
    PipelineTask = PIPECAT_IMPORTS['PipelineTask']
    DailyTransport = PIPECAT_IMPORTS['DailyTransport']
    DailyParams = PIPECAT_IMPORTS['DailyParams']
    TextFrame = PIPECAT_IMPORTS['TextFrame']
    SileroVADAnalyzer = PIPECAT_IMPORTS['SileroVADAnalyzer']
    LLMContext = PIPECAT_IMPORTS['LLMContext']
    
    # Daily.co transport configuration
    print(f"🚗 TRANSPORT: Creating Daily.co transport...")
    print(f"🔗 TRANSPORT: Room URL: {room_url}")
    print(f"🎫 TRANSPORT: Token: {'PROVIDED' if token else 'NONE'}")
    logger.info(f"🚗 Creating Daily.co transport for room: {room_url}")
    
    transport_params = DailyParams(
        audio_out_enabled=True,
        audio_in_enabled=True,
        video_out_enabled=False,
        transcription_enabled=False,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer()
    )
    
    print("🎛️ TRANSPORT: Daily.co parameters:")
    print(f"   - Audio In: {transport_params.audio_in_enabled}")
    print(f"   - Audio Out: {transport_params.audio_out_enabled}")
    print(f"   - Video Out: {transport_params.video_out_enabled}")
    print(f"   - VAD Enabled: {transport_params.vad_enabled}")
    
    transport = DailyTransport(
        room_url=room_url,
        token=token,
        bot_name="Kukma",
        params=transport_params
    )
    
    print("✅ TRANSPORT: Daily.co transport created successfully")
    logger.info("✅ Daily.co transport created")
    
    # Get Pipecat classes from imports
    FrameProcessor = PIPECAT_IMPORTS['FrameProcessor']
    TextFrame = PIPECAT_IMPORTS['TextFrame']
    
    # Create processors for the pipeline
    llm_context = LLMContext()
    
    # Custom processor for cooking responses
    class CookingProcessor(FrameProcessor):
        def __init__(self, assistant: CookingVoiceAssistant):
            super().__init__()
            self.assistant = assistant
            self.frame_count = 0
            print("🎛️ PROCESSOR: CookingProcessor initialized")
            logger.info("🎛️ CookingProcessor initialized")
            
        async def process_frame(self, frame, direction):
            self.frame_count += 1
            frame_type = type(frame).__name__
            
            print(f"🔄 PROCESSOR: Frame #{self.frame_count} - Type: {frame_type}, Direction: {direction}")
            logger.debug(f"🔄 Processing frame #{self.frame_count}: {frame_type} ({direction})")
            
            if isinstance(frame, TextFrame):
                # Process user speech
                user_text = frame.text
                print(f"🎤 PROCESSOR: User speech detected: '{user_text}'")
                logger.info(f"🎤 User said: {user_text}")
                
                # Generate response
                print("🤖 PROCESSOR: Generating AI response...")
                response = await self.assistant.process_user_message(user_text)
                print(f"💭 PROCESSOR: AI response generated: '{response}'")
                logger.info(f"💭 Assistant response: {response}")
                
                # Return response frame
                print("📤 PROCESSOR: Sending response frame to pipeline")
                await self.push_frame(TextFrame(response), direction)
                return
            else:
                print(f"📋 PROCESSOR: Passing through {frame_type} frame")
            
            # Pass through other frames
            await self.push_frame(frame, direction)
    
    cooking_processor = CookingProcessor(assistant)
    
    # Get more Pipecat classes from imports
    Pipeline = PIPECAT_IMPORTS['Pipeline']
    PipelineTask = PIPECAT_IMPORTS['PipelineTask']
    
    # Create pipeline
    print("🔧 PIPELINE: Creating Pipecat pipeline...")
    print("🔧 PIPELINE: Components:")
    print("   1. Daily.co Transport Input")
    print("   2. CookingProcessor (AI)")
    print("   3. Daily.co Transport Output")
    
    pipeline = Pipeline([
        transport.input(),
        cooking_processor,
        transport.output()
    ])
    
    print("✅ PIPELINE: Pipeline created successfully")
    logger.info("✅ Pipecat pipeline created with 3 components")
    
    # Create and return pipeline task
    print("📋 PIPELINE: Creating pipeline task...")
    task = PipelineTask(pipeline)
    print("✅ PIPELINE: Pipeline task created successfully")
    logger.info("✅ Pipeline task created")
    
    return task

# API Endpoints
@app.get("/")
def read_root():
    features = []
    if GEMINI_API_KEY:
        features.append("gemini")
    if PIPECAT_AVAILABLE:
        features.extend(["pipecat", "daily.co"])
    if GROQ_AVAILABLE and GROQ_API_KEY:
        features.append("groq")
    
    return {
        "service": "CookMaa Voice Assistant",
        "status": "running", 
        "version": "2.0.0",
        "features": features,
        "pipecat_available": PIPECAT_AVAILABLE,
        "groq_available": GROQ_AVAILABLE,
        "port": os.getenv("PORT", "8000")
    }

@app.get("/health")
def health_check():
    """Enhanced health check for Railway"""
    try:
        # Basic health indicators
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "CookMaa Voice Assistant",
            "version": "2.0.0",
            "python_version": sys.version.split()[0],
            "working_directory": os.getcwd(),
            "port": os.getenv("PORT", "8000"),
            "features": {
                "gemini_available": bool(GEMINI_API_KEY),
                "groq_available": GROQ_AVAILABLE and bool(GROQ_API_KEY),
                "pipecat_available": PIPECAT_AVAILABLE,
                "daily_available": bool(DAILY_API_KEY)
            }
        }
        
        print(f"🏥 HEALTH: Health check requested at {health_status['timestamp']}")
        return health_status
        
    except Exception as e:
        print(f"❌ HEALTH: Health check failed: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/start-voice-session")
async def start_voice_session(request: VoiceSessionRequest):
    """Start a new voice session with Daily.co + Pipecat"""
    
    print("🎤 SESSION: Voice session request received")
    print(f"📋 SESSION: Recipe context: {bool(request.recipe_context)}")
    print(f"📋 SESSION: Step index: {request.step_index}")
    
    if not PIPECAT_AVAILABLE:
        print("❌ SESSION: Pipecat not available")
        raise HTTPException(
            status_code=503, 
            detail="Pipecat voice pipeline not available. Service running in limited mode."
        )
    
    if not DAILY_API_KEY or not GEMINI_API_KEY:
        print("❌ SESSION: Missing required API keys")
        raise HTTPException(
            status_code=500, 
            detail="Voice session requires DAILY_API_KEY and GEMINI_API_KEY"
        )
    
    try:
        import uuid
        session_id = str(uuid.uuid4())
        
        print(f"🎤 SESSION: Starting voice session {session_id}")
        logger.info(f"🎤 Starting voice session {session_id}")
        
        # Create Daily.co room if not provided
        room_url = request.room_url
        token = request.token
        
        if not room_url or room_url == "temp-room":
            print("🏠 SESSION: Creating new Daily.co room...")
            room_url, token = await create_daily_room()
        else:
            print(f"🏠 SESSION: Using provided room: {room_url}")
            logger.info(f"🏠 Using provided room URL: {room_url}")
        
        print(f"🔗 SESSION: Final room URL: {room_url}")
        
        # Create Pipecat pipeline
        print("🔧 SESSION: Creating Pipecat pipeline...")
        task = await create_pipecat_pipeline(
            room_url=room_url,
            token=token,
            recipe_context=request.recipe_context or {}
        )
        print("✅ SESSION: Pipecat pipeline created")
        
        # Store session
        active_sessions[session_id] = {
            "task": task,
            "room_url": room_url,
            "recipe_context": request.recipe_context,
            "step_index": request.step_index,
            "created_at": datetime.now().isoformat()
        }
        
        print(f"💾 SESSION: Session {session_id} stored in active sessions")
        print(f"📊 SESSION: Total active sessions: {len(active_sessions)}")
        
        # Start the pipeline in background
        print("🚀 SESSION: Starting pipeline in background...")
        asyncio.create_task(run_pipeline(session_id, task))
        
        print(f"✅ SESSION: Voice session {session_id} started successfully")
        logger.info(f"✅ Voice session {session_id} started successfully")
        
        return VoiceSessionResponse(
            status="started",
            session_id=session_id,
            room_url=room_url
        )
        
    except Exception as e:
        error_msg = f"Failed to start voice session: {str(e)}"
        print(f"❌ SESSION: {error_msg}")
        logger.error(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

async def run_pipeline(session_id: str, task):
    """Run Pipecat pipeline for a session"""
    try:
        print(f"🚀 RUNNER: Starting pipeline for session {session_id}")
        logger.info(f"🚀 Running pipeline for session {session_id}")
        
        # Create pipeline runner
        print("🏃 RUNNER: Creating PipelineRunner...")
        PipelineRunner = PIPECAT_IMPORTS['PipelineRunner']
        runner = PipelineRunner()
        print("✅ RUNNER: PipelineRunner created")
        
        # Run the task
        print(f"▶️ RUNNER: Starting pipeline task for session {session_id}")
        print("🎧 RUNNER: Pipeline is now listening for audio input...")
        print("🔊 RUNNER: Pipeline is ready to generate audio output...")
        logger.info(f"▶️ Pipeline task started - listening for audio")
        
        await runner.run(task)
        
        print(f"🏁 RUNNER: Pipeline finished for session {session_id}")
        logger.info(f"🏁 Pipeline finished for session {session_id}")
        
    except Exception as e:
        error_msg = f"Pipeline error for session {session_id}: {str(e)}"
        print(f"❌ RUNNER: {error_msg}")
        logger.error(f"❌ {error_msg}")
        
        # Log additional debug info
        print(f"🔍 RUNNER: Active sessions count: {len(active_sessions)}")
        if session_id in active_sessions:
            session_data = active_sessions[session_id]
            print(f"🔍 RUNNER: Session data keys: {list(session_data.keys())}")
    finally:
        # Clean up session
        if session_id in active_sessions:
            print(f"🧹 RUNNER: Cleaning up session {session_id}")
            del active_sessions[session_id]
            print(f"🧹 RUNNER: Session {session_id} cleaned up")
            logger.info(f"🧹 Cleaned up session {session_id}")
        else:
            print(f"⚠️ RUNNER: Session {session_id} not found in active sessions during cleanup")

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
        "groq_library_available": GROQ_AVAILABLE,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "daily_api_key_configured": bool(DAILY_API_KEY),
        "pipecat_library_available": PIPECAT_AVAILABLE,
        "port": os.getenv("PORT", "not_set"),
        "environment_vars_count": len([k for k in os.environ.keys() if not k.startswith("_")])
    }

@app.get("/debug/groq")
async def test_groq():
    """Test Groq API connectivity (STT/TTS only)"""
    if not GROQ_AVAILABLE:
        return {"error": "Groq library not available"}
    
    if not GROQ_API_KEY:
        return {"error": "No Groq API key configured"}
    
    try:
        # Test basic Groq functionality (we'll use this for STT/TTS)
        return {
            "status": "configured",
            "note": "Groq configured for STT/TTS only",
            "api_key_length": len(GROQ_API_KEY),
            "library_available": GROQ_AVAILABLE
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.post("/{session_id}/connect")
async def connect_rtvi_client(session_id: str, request: dict):
    """RTVI client connection endpoint"""
    
    try:
        print(f"🔗 RTVI: Client connecting to session {session_id}")
        
        if session_id not in active_sessions:
            print(f"❌ RTVI: Session {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_data = active_sessions[session_id]
        room_url = session_data["room_url"]
        
        print(f"🏠 RTVI: Directing client to room: {room_url}")
        
        # Return connection info for the iOS client
        return {
            "room_url": room_url,
            "token": None,  # No token needed for private rooms
            "config": {
                "rtvi": {
                    "voice": "elevenlabs",
                    "llm": "gemini"
                }
            }
        }
        
    except Exception as e:
        error_msg = f"RTVI connection failed: {str(e)}"
        print(f"❌ RTVI: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

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

# Startup completion logging
print("🎉 STARTUP: All initialization completed successfully!")
print("🎉 STARTUP: Application ready to handle requests")
print(f"🎉 STARTUP: Available features: {['gemini' if GEMINI_API_KEY else None, 'pipecat' if PIPECAT_AVAILABLE else None, 'groq' if GROQ_AVAILABLE and GROQ_API_KEY else None]}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 UVICORN: Starting CookMaa backend on port {port}")
    print(f"🚀 UVICORN: Host: 0.0.0.0, Port: {port}")
    print(f"🚀 UVICORN: Datetime: {datetime.now()}")
    print(f"🚀 UVICORN: Health endpoint will be available at: http://0.0.0.0:{port}/health")
    
    try:
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        print(f"❌ UVICORN: Failed to start server: {e}")
        logger.error(f"Failed to start server: {e}")
        raise