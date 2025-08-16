#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# Get VAPI private key from environment (for API management)
VAPI_API_KEY = os.getenv("VAPI_PRIVATE_KEY") or os.getenv("VAPI_API_KEY") or "39a39332-e4d3-412c-9f7f-679ef3963c9f"

if not VAPI_API_KEY:
    print("❌ VAPI_API_KEY not found in environment")
    exit(1)

# Railway webhook URL
WEBHOOK_URL = "https://cookmaa-backend-production.up.railway.app/webhook/vapi-message"

# Complete assistant configuration
assistant_config = {
    "name": "CookMaa Cooking Assistant",
    "transcriber": {
        "provider": "deepgram",
        "model": "nova-2-conversationalai",
        "language": "en",
        "endpointing": 150
    },
    "model": {
        "provider": "openai",
        "model": "gpt-4o",
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": """You are Kukma, a helpful cooking voice assistant for the CookMaa app.

IMPORTANT: Keep responses very brief (1-2 sentences max) for voice interaction.
Be warm, encouraging, and practical. Always relate to the current cooking context.

CURRENT RECIPE CONTEXT:
{{recipe_title}} - Step {{current_step}} of {{total_steps}}
Current Step: {{current_step_text}}
Session ID: {{session_id}}

You help users with:
- Recipe step navigation (next step, repeat, previous step)  
- Cooking tips and guidance for {{recipe_title}}
- Ingredient questions
- Timing and temperature advice
- General cooking encouragement

WAKE WORD DETECTION:
Listen for "Hey Kukma" or "Hey Cookma" followed by commands. When you hear these wake words, process the command that follows.

⚠️ MANDATORY FUNCTION CALLING RULES ⚠️

DO NOT TALK ABOUT CALLING FUNCTIONS - JUST CALL THEM!

When user says these EXACT words, IMMEDIATELY call the function (no conversation):
- "next step" / "next" / "continue" → CALL next_step()
- "repeat" / "repeat step" → CALL repeat_step()  
- "previous" / "go back" → CALL previous_step()

❌ WRONG: "I'll call the next step function for you"
✅ RIGHT: Just call next_step() function immediately

❌ WRONG: "Next step coming up!"
✅ RIGHT: Just call next_step() function immediately

The function returns the actual recipe content - don't narrate, just execute!

For general cooking questions (not step navigation), provide helpful, brief advice.

Examples:
- User: "Hey Kukma, next step" → Call next_step() function immediately
- User: "next step" → Call next_step() function immediately  
- User: "Hey Kukma, how long should I cook this?" → "For this step, about 5-7 minutes should work!"
- User: "repeat" → Call repeat_step() function immediately
"""
            }
        ]
    },
    "voice": {
        "provider": "vapi",
        "voiceId": "Elliot"
    },
    "firstMessage": "Hi! I'm Kukma, ready to help you cook {{recipe_title}}! We're on step {{current_step}} of {{total_steps}}. Say 'Hey Kukma, next step' when you're ready to continue, or ask me any cooking questions!",
    "serverUrl": WEBHOOK_URL,
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
    "endCallMessage": "Thanks for cooking with CookMaa! Hope your dish turns out delicious. Happy cooking!",
    "endCallPhrases": [
        "goodbye",
        "bye", 
        "thank you",
        "that's all",
        "done cooking"
    ],
    "startSpeakingPlan": {
        "waitSeconds": 0.4,
        "smartEndpointingEnabled": True
    },
    "functions": [
        {
            "name": "next_step",
            "description": "Move to the next step in the cooking recipe",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "repeat_step",
            "description": "Repeat the current cooking step instructions",
            "parameters": {
                "type": "object", 
                "properties": {},
                "required": []
            }
        },
        {
            "name": "previous_step",
            "description": "Go back to the previous cooking step",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]
}

def create_assistant():
    """Create a new VAPI assistant using their API"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print("🤖 Creating CookMaa VAPI Assistant...")
    print(f"📡 Webhook URL: {WEBHOOK_URL}")
    
    try:
        response = requests.post(
            "https://api.vapi.ai/assistant",
            headers=headers,
            json=assistant_config,
            timeout=30
        )
        
        if response.status_code == 201:
            assistant = response.json()
            assistant_id = assistant.get("id")
            
            print("✅ Assistant created successfully!")
            print(f"🆔 Assistant ID: {assistant_id}")
            print(f"📋 Name: {assistant.get('name')}")
            print(f"🎤 First Message: {assistant.get('firstMessage')}")
            print(f"📡 Server URL: {assistant.get('serverUrl')}")
            
            # Save assistant ID for future reference
            with open("vapi_assistant_id.txt", "w") as f:
                f.write(assistant_id)
            
            print(f"💾 Assistant ID saved to vapi_assistant_id.txt")
            
            return assistant_id
            
        else:
            print(f"❌ Failed to create assistant: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error creating assistant: {str(e)}")
        return None

def update_existing_assistant(assistant_id):
    """Update existing assistant with new configuration"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"🔄 Updating existing assistant {assistant_id}...")
    
    try:
        response = requests.patch(
            f"https://api.vapi.ai/assistant/{assistant_id}",
            headers=headers,
            json=assistant_config,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Assistant updated successfully!")
            return True
        else:
            print(f"❌ Failed to update assistant: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating assistant: {str(e)}")
        return False

if __name__ == "__main__":
    print("🎯 VAPI Assistant Setup")
    print("=" * 50)
    
    # Update existing assistant with functions
    existing_id = "e45d3c24-b950-44dc-8ba6-a7632218365c"
    print(f"🔄 Updating assistant {existing_id} with function definitions...")
    
    if update_existing_assistant(existing_id):
        print(f"\n🎉 Success! Updated assistant: {existing_id}")
        print("✅ Functions added: next_step, repeat_step, previous_step")
    else:
        print("❌ Failed to update assistant")