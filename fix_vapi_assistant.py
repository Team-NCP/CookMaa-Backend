#!/usr/bin/env python3

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

# VAPI configuration
VAPI_API_KEY = os.getenv("VAPI_PRIVATE_KEY") or "39a39332-e4d3-412c-9f7f-679ef3963c9f"
ASSISTANT_ID = "b9c6dfa6-d816-4af9-b5e8-ac924baf6509"
WEBHOOK_URL = "https://cookmaa-backend-production.up.railway.app/webhook/vapi-message"

def fix_assistant_configuration():
    """Remove deprecated functions and ensure proper Tools configuration"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Get current assistant configuration
    print(f"🔍 Getting current assistant configuration...")
    
    try:
        response = requests.get(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ Failed to get assistant: {response.status_code} - {response.text}")
            return False
            
        current_config = response.json()
        print(f"✅ Current assistant retrieved")
        
        # Build clean configuration without deprecated functions
        clean_config = {
            "name": current_config.get("name", "CookMaa Cooking Assistant"),
            "transcriber": current_config.get("transcriber", {}),
            "model": {
                "provider": "openai",
                "model": "gpt-4o-mini", 
                "temperature": 0.0,
                "messages": [{
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

TOOL USAGE:
When users say step navigation commands, use the available tools immediately:
- "next step" or "next" → Use next_step tool
- "repeat" or "repeat step" → Use repeat_step tool  
- "previous" or "go back" → Use previous_step tool

Do not announce tool usage. Let the tool response speak for itself.

For general cooking questions (not step navigation), provide helpful, brief advice.

Examples:
- User: "next step" → [Call next_step()]
- User: "repeat" → [Call repeat_step()]  
- User: "how long to cook?" → "About 5-7 minutes should work!"
"""
                }]
                # Remove the deprecated functions array entirely
                # Tools are managed via VAPI dashboard
            },
            "voice": current_config.get("voice", {}),
            "firstMessage": "Hi! I'm Kukma, ready to help you cook {{recipe_title}}! We're on step {{current_step}} of {{total_steps}}. Say 'Hey Kukma, next step' when you're ready to continue, or ask me any cooking questions!",
            "serverUrl": WEBHOOK_URL,
            "serverMessages": ["transcript", "function-call", "hang"],
            "clientMessages": ["conversation-update", "function-call", "hang", "speech-update", "transcript"],
            "backgroundDenoisingEnabled": True,
            "endCallMessage": "Thanks for cooking with CookMaa! Hope your dish turns out delicious. Happy cooking!",
            "endCallPhrases": ["goodbye", "bye", "thank you", "that's all", "done cooking"],
            "startSpeakingPlan": {
                "waitSeconds": 0.4,
                "smartEndpointingEnabled": True
            }
        }
        
        print(f"🧹 Updating assistant with clean configuration (no deprecated functions)...")
        
        # Update the assistant
        response = requests.patch(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            json=clean_config,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Assistant updated successfully!")
            print("🔧 Removed deprecated functions array")
            print("📡 Webhook URL confirmed")
            print("🛠️ Tools should now work via dashboard configuration")
            return True
        else:
            print(f"❌ Failed to update assistant: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing assistant: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 VAPI Assistant Fix Tool")
    print("=" * 50)
    
    success = fix_assistant_configuration()
    
    if success:
        print(f"\n🎉 Assistant {ASSISTANT_ID} has been fixed!")
        print("✅ Deprecated functions removed")
        print("✅ Clean Tools-only configuration")
        print("✅ Function calls should now reach webhook")
        print("\n📝 Next steps:")
        print("1. Test voice commands in the app")
        print("2. Check that 'next step' commands trigger function calls")
        print("3. Verify webhook receives function-call messages")
    else:
        print("❌ Failed to fix assistant configuration")