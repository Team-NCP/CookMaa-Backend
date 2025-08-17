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
        
        # Handle trigger messages from iOS for auto-announcements
        if "start cooking session" in user_text_lower:
            return await self.handle_welcome_announcement()
        elif user_text_lower.startswith("read step "):
            try:
                step_num = int(user_text_lower.replace("read step ", ""))
                return await self.handle_step_announcement(step_num)
            except ValueError:
                pass
        
        # Handle step navigation commands
        elif "next step" in user_text_lower:
            return await self.handle_next_step()
        elif "repeat" in user_text_lower and "step" in user_text_lower:
            return await self.handle_repeat_step()
        elif "previous step" in user_text_lower or "go back" in user_text_lower:
            return await self.handle_previous_step()
        else:
            # Handle general conversation with Gemini
            return await self.handle_conversation(user_text)
    
    async def handle_next_step(self) -> Dict[str, Any]:
        """Move to next recipe step and announce it"""
        
        if self.recipe.step_index + 1 < len(self.recipe.steps):
            self.recipe.step_index += 1
            current_step = self.recipe.steps[self.recipe.step_index]
            
            # Create a natural conversational response with full step content
            response = f"Great! Moving to step {self.recipe.step_index + 1} of {len(self.recipe.steps)}. {current_step}. Take your time with this step!"
            
            logger.info(f"Moved to step {self.recipe.step_index + 1}: {current_step}")
            
            return {
                "response": response,
                "action": "step_changed",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            return {
                "response": "Excellent! You've completed all the steps for this recipe. Your delicious dish should be ready to serve!",
                "action": "recipe_completed",
                "step_index": self.recipe.step_index
            }
    
    async def handle_repeat_step(self) -> Dict[str, Any]:
        """Repeat current recipe step"""
        
        if self.recipe.steps and self.recipe.step_index < len(self.recipe.steps):
            current_step = self.recipe.steps[self.recipe.step_index]
            response = f"Let me repeat step {self.recipe.step_index + 1} for you. {current_step}. Got it?"
            
            logger.info(f"Repeating step {self.recipe.step_index + 1}")
            
            return {
                "response": response,
                "action": "step_repeated",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            return {
                "response": "I don't have a current step to repeat. Let me know if you need help with anything else!",
                "action": "no_step"
            }
    
    async def handle_previous_step(self) -> Dict[str, Any]:
        """Move to previous recipe step and announce it"""
        
        if self.recipe.step_index > 0:
            self.recipe.step_index -= 1
            current_step = self.recipe.steps[self.recipe.step_index]
            
            response = f"Going back to step {self.recipe.step_index + 1}. {current_step}. You're doing great!"
            
            logger.info(f"Moved back to step {self.recipe.step_index + 1}: {current_step}")
            
            return {
                "response": response,
                "action": "step_changed",
                "step_index": self.recipe.step_index,
                "step_text": current_step
            }
        else:
            first_step = self.recipe.steps[0] if self.recipe.steps else "No steps available"
            response = f"You're already at the first step. Let me repeat it: {first_step}. Ready to continue?"
            
            return {
                "response": response,
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
    
    async def handle_welcome_announcement(self) -> Dict[str, Any]:
        """Handle welcome message and first step announcement"""
        
        recipe_title = self.recipe.title if hasattr(self.recipe, 'title') else "this recipe"
        first_step = self.recipe.steps[0] if self.recipe.steps else "No steps available"
        total_steps = len(self.recipe.steps)
        
        welcome_message = f"Hello! I'm Kukma, your cooking assistant. Let's cook {recipe_title} together! This recipe has {total_steps} steps. Here's step 1: {first_step}"
        
        logger.info(f"Welcome announcement: {welcome_message}")
        
        return {
            "response": welcome_message,
            "action": "welcome_announced",
            "step_index": 0
        }
    
    async def handle_step_announcement(self, step_num: int) -> Dict[str, Any]:
        """Handle step announcement for specific step number"""
        
        if step_num <= 0 or step_num > len(self.recipe.steps):
            return {
                "response": "Invalid step number.",
                "action": "error"
            }
        
        step_index = step_num - 1
        current_step = self.recipe.steps[step_index]
        
        announcement = f"Step {step_num}: {current_step}"
        
        logger.info(f"Step announcement: {announcement}")
        
        return {
            "response": announcement,
            "action": "step_announced",
            "step_index": step_index
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
        
        # Extract headers for debugging
        headers = dict(request.headers)
        
        print(f"🎤 VAPI Webhook received: {json.dumps(payload, indent=2)}")
        print(f"📋 Request headers: {json.dumps(headers, indent=2)}")
        logger.info(f"VAPI webhook payload received")
        
        # Log all webhook calls for debugging with more detail
        with open("/tmp/vapi_calls.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}: HEADERS={json.dumps(headers)} PAYLOAD={json.dumps(payload)}\n")
        
        # Extract message details
        message_type = payload.get("message", {}).get("type", "")
        
        # Check if this is a direct tool test call (has X-VAPI-Tool header)
        headers = dict(request.headers)
        if "x-vapi-tool" in headers:
            tool_name = headers["x-vapi-tool"]
            print(f"🔧 Direct tool test call for: {tool_name}")
            print(f"🔧 Tool payload: {json.dumps(payload, indent=2)}")
            
            # Handle direct tool calls by simulating function call format
            fake_payload = {
                "message": {
                    "type": "function-call",
                    "functionCall": {
                        "name": tool_name,
                        "arguments": {}
                    }
                },
                "call": {
                    "id": "tool-test-call"
                }
            }
            return await handle_function_call(fake_payload)
        
        if message_type == "function-call":
            # Handle function calls (if using VAPI functions)
            return await handle_function_call(payload)
            
        elif message_type == "tool-calls":
            # Handle VAPI tool calls (this is the format VAPI actually sends)
            print(f"🔧 Processing tool-calls message type")
            return await handle_tool_calls(payload)
            
        elif message_type == "transcript":
            # Handle user speech transcript - process step navigation commands here
            return await handle_transcript(payload)
            
        elif message_type == "hang":
            # Handle call end
            return await handle_call_end(payload)
            
        else:
            # Unknown message type - log the full payload for debugging
            logger.warning(f"Unknown VAPI message type: '{message_type}'")
            logger.warning(f"Full payload: {json.dumps(payload, indent=2)}")
            
            # Check if this might be a tool call with different format
            if "toolCall" in json.dumps(payload) or "function" in json.dumps(payload):
                print(f"🔧 Possible tool call with unknown format - attempting to handle")
                
                # Try to extract function name from anywhere in the payload
                payload_str = json.dumps(payload).lower()
                if "next_step" in payload_str:
                    tool_name = "next_step"
                elif "repeat_step" in payload_str:
                    tool_name = "repeat_step"
                elif "previous_step" in payload_str:
                    tool_name = "previous_step"
                else:
                    return {"message": "Unknown message type"}
                
                # Create fake function call format
                fake_payload = {
                    "message": {
                        "type": "function-call",
                        "functionCall": {
                            "name": tool_name,
                            "arguments": {}
                        }
                    },
                    "call": {
                        "id": payload.get("call", {}).get("id", "unknown-call")
                    }
                }
                return await handle_function_call(fake_payload)
            
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
        
        # Use call_id as session identifier
        session_id = call_id
        
        # Link existing cooking session to this call if not already linked
        if session_id not in cooking_sessions:
            # Look for existing cooking session to link to this call
            for existing_session_id, session in list(cooking_sessions.items()):
                if existing_session_id != call_id and session.recipe_context:
                    # Link this call to the existing session
                    cooking_sessions[call_id] = session
                    session_id = call_id
                    print(f"🔗 Linked VAPI call {call_id} to cooking session {existing_session_id}")
                    break
        
        # If still no session found, return error
        if session_id not in cooking_sessions:
            return {"message": "No active cooking session found. Please start cooking from a recipe in the app first."}
        
        # Process the message with cooking assistant
        session = cooking_sessions[session_id]
        assistant = CookingAssistant(session)
        
        result = await assistant.process_message(transcript)
        
        response_text = result.get("response", "I didn't understand that.")
        
        print(f"🤖 Kukma responds: '{response_text}'")
        logger.info(f"Generated response for transcript: {response_text}")
        
        # Return response for VAPI to speak (simple string format)
        return response_text
        
    except Exception as e:
        logger.error(f"❌ Transcript handling error: {str(e)}")
        return {"message": "Sorry, I had trouble processing that."}

async def handle_tool_calls(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle VAPI tool-calls message type"""
    
    try:
        tool_calls = payload.get("message", {}).get("toolCalls", [])
        if not tool_calls:
            return {"result": "No tool calls found"}
        
        # Process the first tool call
        tool_call = tool_calls[0]
        function_name = tool_call.get("function", {}).get("name", "")
        tool_call_id = tool_call.get("id", "")
        call_id = payload.get("call", {}).get("id", "unknown-call")
        chat_id = payload.get("chat", {}).get("id", "")
        
        print(f"🔧 Tool call: {function_name} (ID: {tool_call_id}) for call: {call_id}, chat: {chat_id}")
        
        # Convert to function-call format and handle
        function_call_payload = {
            "message": {
                "type": "function-call",
                "functionCall": {
                    "name": function_name,
                    "arguments": tool_call.get("function", {}).get("arguments", {})
                }
            },
            "call": {
                "id": call_id
            },
            "chat": {
                "id": chat_id
            }
        }
        
        # Process with existing function call handler
        return await handle_function_call(function_call_payload)
        
    except Exception as e:
        logger.error(f"❌ Tool calls handling error: {str(e)}")
        return {"result": "Sorry, I had trouble with that command."}

async def handle_function_call(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Handle VAPI function calls with proper cooking logic"""
    
    try:
        function_call = payload.get("message", {}).get("functionCall", {})
        function_name = function_call.get("name", "")
        call_id = payload.get("call", {}).get("id", "")
        chat_id = payload.get("chat", {}).get("id", "")
        
        # Use chat_id as primary identifier, fallback to call_id
        session_identifier = chat_id if chat_id else call_id
        
        print(f"🔧 Function call: {function_name} for call: {call_id}, chat: {chat_id}, using: {session_identifier}")
        
        # Get existing cooking session for this call
        session_id = session_identifier
        if session_id not in cooking_sessions:
            # Look for any active cooking session to link to this call
            print(f"🔍 Session {session_identifier} not found. Available sessions: {list(cooking_sessions.keys())}")
            
            # Find the most recent session (likely the active one)
            if cooking_sessions:
                # Get the latest session by finding the one with the most recent timestamp
                latest_session_id = max(cooking_sessions.keys(), 
                                      key=lambda k: cooking_sessions[k].created_at if hasattr(cooking_sessions[k], 'created_at') else '')
                session = cooking_sessions[latest_session_id]
                
                # Link this call to the latest session
                cooking_sessions[session_identifier] = session
                session_id = session_identifier
                print(f"🔗 Linked VAPI session {session_identifier} to latest cooking session {latest_session_id}")
            else:
                print(f"❌ No active cooking session found for session {session_identifier}")
                return {"result": "No active cooking session found. Please start cooking from a recipe in the app first."}
        
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
        
        # Return the function result for VAPI - use clean single format with full content
        response_text = result.get("response", "Function executed")
        response_data = {"result": response_text}
        
        print(f"🔧 Returning function response: {response_text}")
        print(f"🔧 Full response data: {json.dumps(response_data, indent=2)}")
        
        # Log the response for debugging
        with open("/tmp/vapi_responses.log", "a") as f:
            f.write(f"{datetime.now().isoformat()}: FUNCTION_RESPONSE={json.dumps(response_data)}\n")
        
        return response_data
        
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

@app.post("/announce-step")
async def announce_step(request: Request):
    """Send announcement through VAPI by returning it as a response"""
    
    try:
        payload = await request.json()
        session_id = payload.get("session_id")
        message = payload.get("message")
        
        print(f"📢 Announcement request for session {session_id}: {message}")
        logger.info(f"Announcing: {message}")
        
        # Return the message in a format that can be used by VAPI
        return {
            "status": "success",
            "message": message,
            "speak": message  # This will be used for TTS
        }
        
    except Exception as e:
        logger.error(f"❌ Announcement error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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

@app.get("/debug/vapi-calls")
async def get_vapi_calls():
    """Get recent VAPI webhook calls for debugging"""
    
    try:
        with open("/tmp/vapi_calls.log", "r") as f:
            lines = f.readlines()
            recent_calls = lines[-10:]  # Last 10 calls
        
        return {
            "recent_calls_count": len(recent_calls),
            "recent_calls": [line.strip() for line in recent_calls]
        }
    except FileNotFoundError:
        return {
            "recent_calls_count": 0,
            "recent_calls": [],
            "message": "No VAPI calls logged yet"
        }

@app.get("/debug/vapi-responses")
async def get_vapi_responses():
    """Get recent VAPI webhook responses for debugging"""
    
    try:
        with open("/tmp/vapi_responses.log", "r") as f:
            lines = f.readlines()
            recent_responses = lines[-10:]  # Last 10 responses
        
        return {
            "recent_responses_count": len(recent_responses),
            "recent_responses": [line.strip() for line in recent_responses]
        }
    except FileNotFoundError:
        return {
            "recent_responses_count": 0,
            "recent_responses": [],
            "message": "No VAPI responses logged yet"
        }

@app.get("/debug/live-logs")
async def get_live_logs():
    """Get both calls and responses for debugging"""
    
    calls_data = await get_vapi_calls()
    responses_data = await get_vapi_responses()
    
    return {
        "calls": calls_data,
        "responses": responses_data,
        "timestamp": datetime.now().isoformat()
    }

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