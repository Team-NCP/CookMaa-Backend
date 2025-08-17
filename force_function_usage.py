#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

VAPI_API_KEY = os.getenv("VAPI_PRIVATE_KEY") or "39a39332-e4d3-412c-9f7f-679ef3963c9f"
ASSISTANT_ID = "b9c6dfa6-d816-4af9-b5e8-ac924baf6509"

def force_function_usage():
    """Update assistant with more aggressive function calling instructions"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Ultra-aggressive system prompt that forces function usage
    aggressive_config = {
        "name": "CookMaa Cooking Assistant",
        "transcriber": {
            "model": "nova-2-conversationalai",
            "language": "en",
            "endpointing": 150,
            "provider": "deepgram"
        },
        "model": {
            "messages": [
                {
                    "content": """You are Kukma, a cooking assistant. You MUST call functions for navigation commands.

CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

When user says ANY of these phrases, immediately call the specified function:
- "next step", "next", "continue", "move on" → CALL next_step() 
- "repeat", "repeat step", "say that again" → CALL repeat_step()
- "previous", "go back", "last step" → CALL previous_step()

DO NOT explain, DO NOT acknowledge - just call the function immediately.

For non-navigation questions (like "how long to cook?"), give brief helpful advice.

Examples:
User: "next step" → [You MUST call next_step() function]
User: "repeat that" → [You MUST call repeat_step() function]  
User: "how hot should the oil be?" → "Medium heat should work perfectly!"

NEVER say "I'll help you with the next step" - just call the function.""",
                    "role": "system"
                }
            ],
            "model": "gpt-4o-mini",
            "temperature": 0.0,
            "provider": "openai"
        },
        "voice": {
            "voiceId": "Elliot",
            "provider": "vapi"
        },
        "firstMessage": "Ready to cook! Say 'next step' to continue, 'repeat' to hear the current step again, or ask me any cooking questions!",
        "serverUrl": "https://cookmaa-backend-production.up.railway.app/webhook/vapi-message",
        "serverMessages": [
            "transcript",
            "function-call",
            "hang"
        ],
        "clientMessages": [
            "conversation-update",
            "function-call", 
            "hang",
            "speech-update",
            "transcript"
        ],
        "backgroundDenoisingEnabled": True,
        "endCallMessage": "Thanks for cooking with CookMaa! Hope your dish turns out delicious!",
        "endCallPhrases": [
            "goodbye",
            "bye", 
            "thank you",
            "that's all"
        ],
        "startSpeakingPlan": {
            "waitSeconds": 0.4,
            "smartEndpointingEnabled": True
        },
        "functions": [
            {
                "name": "next_step",
                "description": "MANDATORY: Call this when user says 'next step', 'next', 'continue', 'move on', 'what's next'. Do not explain, just call this function.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "repeat_step", 
                "description": "MANDATORY: Call this when user says 'repeat', 'repeat step', 'say that again', 'what was that', 'tell me again'. Do not explain, just call this function.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "previous_step",
                "description": "MANDATORY: Call this when user says 'previous', 'go back', 'last step', 'previous step', 'back'. Do not explain, just call this function.",
                "parameters": {
                    "type": "object", 
                    "properties": {},
                    "required": []
                }
            }
        ]
    }
    
    print("🚀 Applying ultra-aggressive function calling configuration...")
    
    try:
        response = requests.patch(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            json=aggressive_config,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Assistant updated with aggressive function calling!")
            print("🎯 Key changes:")
            print("   - Temperature set to 0.0 (most deterministic)")
            print("   - System prompt demands immediate function calls")
            print("   - Functions marked as MANDATORY")
            print("   - Removed all explanatory language")
            
            return True
        else:
            print(f"❌ Failed to update: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Forcing Function Usage in VAPI Assistant")
    print("=" * 50)
    
    success = force_function_usage()
    
    if success:
        print(f"\n🎉 Assistant updated successfully!")
        print("\n🧪 Test these exact phrases:")
        print("   - 'Hey Kukma, next step'")
        print("   - 'Hey Kukma, repeat'") 
        print("   - 'Hey Kukma, go back'")
        print("\n📊 The assistant should now IMMEDIATELY call functions")
        print("   instead of giving conversational responses!")
    else:
        print("\n❌ Failed to update assistant")