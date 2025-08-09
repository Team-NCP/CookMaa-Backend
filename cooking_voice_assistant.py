#!/usr/bin/env python3

import os
import logging
import json
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Gemini for LLM
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini imported successfully")
except ImportError as e:
    GEMINI_AVAILABLE = False
    print(f"❌ Gemini import error: {e}")

# API Configuration  
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VAPI_API_KEY = os.getenv("VAPI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("✅ Gemini configured with API key")
else:
    print("❌ GEMINI_API_KEY not found")

# FastAPI App
app = FastAPI(title="CookMaa VAPI Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models
class RecipeContext(BaseModel):
    title: str
    steps: List[str]
    step_index: int = 0
    ingredients: Optional[List[str]] = None
    total_steps: Optional[int] = None

class CookingSession(BaseModel):
    session_id: str
    recipe_context: RecipeContext
    user_preferences: Dict[str, Any] = {}
    conversation_history: List[Dict[str, str]] = []
    created_at: str

class VAPIWebhookRequest(BaseModel):
    message: Dict[str, Any]
    call: Dict[str, Any]
    
# Global session storage (In production, use Redis or Database)
cooking_sessions: Dict[str, CookingSession] = {}

class CookingAssistant:
    """Cooking assistant powered by Gemini 1.5 Flash"""
    
    def __init__(self, session: CookingSession):
        self.session = session
        self.recipe = session.recipe_context
        
    def build_system_prompt(self) -> str:
        """Build context-aware system prompt for Gemini"""
        
        current_step_index = self.recipe.step_index
        current_step = "No recipe loaded"
        
        if self.recipe.steps and current_step_index < len(self.recipe.steps):
            current_step = self.recipe.steps[current_step_index]
            
        system_prompt = f"""You are Kukma, a helpful cooking voice assistant for the CookMaa app.

IMPORTANT: Keep responses very brief (1-2 sentences max) for voice interaction.
Be warm, encouraging, and practical. Always relate to the current cooking context.

Current Recipe: {self.recipe.title}
Current Step ({current_step_index + 1}/{len(self.recipe.steps)}): {current_step}

Voice Commands you should handle:
- "next step" or "go to next step" → Move to step {current_step_index + 2} and read it
- "repeat" or "repeat step" → Repeat the current step instruction  
- "previous step" or "go back" → Move to step {current_step_index} and read it
- Recipe questions → Answer based on current step context

Always be encouraging and helpful with cooking guidance!"""

        return system_prompt
    
    async def process_message(self, user_text: str) -> Dict[str, Any]:
        """Process user message and return response + actions"""
        
        user_text_lower = user_text.lower().strip()
        
        # Handle step navigation commands
        if "next step" in user_text_lower:
            return await self.handle_next_step()
        elif "repeat" in user_text_lower and "step" in user_text_lower:
            return await self.handle_repeat_step()
        elif "previous step" in user_text_lower or "go back" in user_text_lower:
            return await self.handle_previous_step()
        else:
            # Handle general conversation with Gemini
            return await self.handle_conversation(user_text)
    
    async def handle_next_step(self) -> Dict[str, Any]:
        """Move to next recipe step"""
        
        if self.recipe.step_index + 1 < len(self.recipe.steps):
            self.recipe.step_index += 1
            current_step = self.recipe.steps[self.recipe.step_index]
            
            response = f"Step {self.recipe.step_index + 1}: {current_step}"
            
            logger.info(f"Moved to step {self.recipe.step_index + 1}")
            
            return {
                "response": response,
                "action": "step_changed",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            return {
                "response": "Great job! You've completed all the steps for this recipe. Your dish should be ready!",
                "action": "recipe_completed",
                "step_index": self.recipe.step_index
            }
    
    async def handle_repeat_step(self) -> Dict[str, Any]:
        """Repeat current recipe step"""
        
        if self.recipe.steps and self.recipe.step_index < len(self.recipe.steps):
            current_step = self.recipe.steps[self.recipe.step_index]
            response = f"Step {self.recipe.step_index + 1}: {current_step}"
            
            return {
                "response": response,
                "action": "step_repeated",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            return {
                "response": "No recipe step to repeat right now.",
                "action": "no_step"
            }
    
    async def handle_previous_step(self) -> Dict[str, Any]:
        """Move to previous recipe step"""
        
        if self.recipe.step_index > 0:
            self.recipe.step_index -= 1
            current_step = self.recipe.steps[self.recipe.step_index]
            
            response = f"Going back to step {self.recipe.step_index + 1}: {current_step}"
            
            logger.info(f"Moved back to step {self.recipe.step_index + 1}")
            
            return {
                "response": response,
                "action": "step_changed",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            return {
                "response": "You're already at the first step of the recipe.",
                "action": "first_step",
                "step_index": 0
            }
    
    async def handle_conversation(self, user_text: str) -> Dict[str, Any]:
        """Handle general conversation with Gemini"""
        
        try:
            system_prompt = self.build_system_prompt()
            
            # Build conversation context
            conversation_text = f"{system_prompt}\n\nUser: {user_text}"
            
            # Add recent conversation history for context
            if self.session.conversation_history:
                history_context = "\n".join([
                    f"{msg['role'].title()}: {msg['content']}" 
                    for msg in self.session.conversation_history[-4:]  # Last 4 exchanges
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
                return {
                    "response": "I'm sorry, I didn't catch that. Could you try again?",
                    "action": "clarification_needed"
                }
            
            response_text = response.text.strip()
            
            # Update conversation history
            self.session.conversation_history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": response_text}
            ])
            
            # Keep only recent history to avoid token limits
            if len(self.session.conversation_history) > 8:
                self.session.conversation_history = self.session.conversation_history[-8:]
            
            logger.info(f"💭 Gemini response: {response_text}")
            
            return {
                "response": response_text,
                "action": "conversation",
                "step_index": self.recipe.step_index
            }
            
        except Exception as e:
            logger.error(f"❌ Gemini LLM processing error: {str(e)}")
            return {
                "response": "I'm having trouble thinking right now. Could you repeat that?",
                "action": "error",
                "error": str(e)
            }

# API Endpoints
@app.get("/")
def read_root():
    return {
        "service": "CookMaa VAPI Assistant",
        "status": "running",
        "version": "1.0.0",
        "gemini_available": GEMINI_AVAILABLE
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "gemini": bool(GEMINI_API_KEY),
            "vapi": bool(VAPI_API_KEY)
        }
    }

@app.post("/create-cooking-session")
async def create_cooking_session(recipe_context: RecipeContext):
    """Create a new cooking session with recipe context"""
    
    import uuid
    session_id = str(uuid.uuid4())
    
    session = CookingSession(
        session_id=session_id,
        recipe_context=recipe_context,
        created_at=datetime.now().isoformat()
    )
    
    cooking_sessions[session_id] = session
    
    logger.info(f"Created cooking session {session_id} for recipe: {recipe_context.title}")
    
    return {
        "session_id": session_id,
        "recipe_title": recipe_context.title,
        "total_steps": len(recipe_context.steps),
        "current_step": recipe_context.step_index + 1
    }

@app.post("/webhook/vapi-message")
async def vapi_webhook(request: Request):
    """Handle VAPI webhook messages"""
    
    try:
        # Parse the webhook payload
        payload = await request.json()
        
        print(f"🎤 VAPI Webhook received: {json.dumps(payload, indent=2)}")
        logger.info(f"VAPI webhook payload received")
        
        # Extract message details
        message_type = payload.get("message", {}).get("type", "")
        
        if message_type == "function-call":
            # Handle function calls (if using VAPI functions)
            return await handle_function_call(payload)
            
        elif message_type == "transcript":
            # Handle user speech transcript
            return await handle_transcript(payload)
            
        elif message_type == "hang":
            # Handle call end
            return await handle_call_end(payload)
            
        else:
            # Unknown message type
            logger.warning(f"Unknown VAPI message type: {message_type}")
            return {"message": "Unknown message type"}
            
    except Exception as e:
        logger.error(f"❌ VAPI webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def handle_transcript(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle user speech transcript from VAPI"""
    
    try:
        # Extract user transcript
        transcript = payload.get("message", {}).get("transcript", "")
        call_id = payload.get("call", {}).get("id", "")
        
        if not transcript:
            return {"message": "No transcript found"}
        
        print(f"🎤 User said: '{transcript}'")
        
        # For now, use a default session (in production, map call_id to session_id)
        session_id = call_id  # or extract from call metadata
        
        # Get or create a default session for testing
        if session_id not in cooking_sessions:
            # Create a default recipe session for testing
            default_recipe = RecipeContext(
                title="Test Recipe",
                steps=[
                    "Heat oil in a pan",
                    "Add onions and sauté",
                    "Add spices and cook",
                    "Add vegetables",
                    "Cook until done"
                ],
                step_index=0
            )
            
            cooking_sessions[session_id] = CookingSession(
                session_id=session_id,
                recipe_context=default_recipe,
                created_at=datetime.now().isoformat()
            )
        
        # Process the message with cooking assistant
        session = cooking_sessions[session_id]
        assistant = CookingAssistant(session)
        
        result = await assistant.process_message(transcript)
        
        response_text = result.get("response", "I didn't understand that.")
        
        print(f"🤖 Kukma responds: '{response_text}'")
        logger.info(f"Generated response for transcript: {response_text}")
        
        # Return response for VAPI to speak
        return {
            "message": response_text,
            "action": result.get("action"),
            "step_index": result.get("step_index")
        }
        
    except Exception as e:
        logger.error(f"❌ Transcript handling error: {str(e)}")
        return {"message": "Sorry, I had trouble processing that."}

async def handle_function_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle VAPI function calls with proper cooking logic"""
    
    try:
        function_call = payload.get("message", {}).get("functionCall", {})
        function_name = function_call.get("name", "")
        call_id = payload.get("call", {}).get("id", "")
        
        print(f"🔧 Function call: {function_name} for call: {call_id}")
        
        # Get or create session for this call
        session_id = call_id
        if session_id not in cooking_sessions:
            # Create a default recipe session
            default_recipe = RecipeContext(
                title="Default Recipe",
                steps=[
                    "Heat oil in a pan over medium heat",
                    "Add onions and sauté until golden brown",
                    "Add spices and cook for 1 minute",
                    "Add main ingredients and cook",
                    "Season and serve hot"
                ],
                step_index=0
            )
            
            cooking_sessions[session_id] = CookingSession(
                session_id=session_id,
                recipe_context=default_recipe,
                created_at=datetime.now().isoformat()
            )
        
        # Get session and process function call
        session = cooking_sessions[session_id]
        assistant = CookingAssistant(session)
        
        # Handle different function calls
        if function_name == "next_step":
            result = await assistant.handle_next_step()
        elif function_name == "repeat_step":
            result = await assistant.handle_repeat_step()
        elif function_name == "previous_step":
            result = await assistant.handle_previous_step()
        else:
            return {"result": f"Unknown function: {function_name}"}
        
        # Return the function result for VAPI
        return {
            "result": result.get("response", "Function executed"),
            "action": result.get("action"),
            "step_index": result.get("step_index")
        }
        
    except Exception as e:
        logger.error(f"❌ Function call error: {str(e)}")
        return {"result": "Sorry, I had trouble with that command."}

async def handle_call_end(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle call end"""
    
    call_id = payload.get("call", {}).get("id", "")
    
    print(f"📞 Call ended: {call_id}")
    logger.info(f"Call ended: {call_id}")
    
    # Clean up session if needed
    if call_id in cooking_sessions:
        del cooking_sessions[call_id]
        logger.info(f"Cleaned up session: {call_id}")
    
    return {"message": "Call ended"}

@app.get("/sessions")
async def get_active_sessions():
    """Get active cooking sessions"""
    
    sessions = []
    for session_id, session in cooking_sessions.items():
        sessions.append({
            "session_id": session_id,
            "recipe_title": session.recipe_context.title,
            "current_step": session.recipe_context.step_index + 1,
            "total_steps": len(session.recipe_context.steps)
        })
    
    return {"active_sessions": sessions, "count": len(sessions)}

@app.get("/debug/test-gemini")
async def test_gemini():
    """Test Gemini API connectivity"""
    
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return {"error": "Gemini not configured"}
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            "Say hello as Kukma, a cooking assistant, in one brief sentence.",
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=50
            )
        )
        
        return {
            "status": "success",
            "model": "gemini-1.5-flash",
            "response": response.text,
            "usage": "LLM for cooking conversation"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"🚀 Starting CookMaa VAPI Assistant on port {port}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )