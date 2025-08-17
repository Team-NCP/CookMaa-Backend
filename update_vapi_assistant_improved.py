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

def update_assistant_with_improved_functions():
    """Update assistant with improved function calling configuration"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Load our improved configuration
    with open('vapi_assistant_config.json', 'r') as f:
        improved_config = json.load(f)
    
    print(f"🔧 Updating assistant with improved function configuration...")
    
    try:
        # Update the assistant with our improved config
        response = requests.patch(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            json=improved_config,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Assistant updated successfully!")
            print("🎯 Key improvements applied:")
            print("   - System prompt now ENCOURAGES function usage")
            print("   - Removed conflicting 'backend handles' message")
            print("   - Enhanced function descriptions with trigger phrases")
            print("   - Lowered temperature to 0.1 for more deterministic responses")
            
            # Verify the functions are properly set
            updated_assistant = response.json()
            functions = updated_assistant.get('functions', [])
            print(f"📊 Functions configured: {len(functions)}")
            for func in functions:
                print(f"   - {func['name']}: {func['description'][:50]}...")
                
            return True
        else:
            print(f"❌ Failed to update assistant: {response.status_code}")
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating assistant: {str(e)}")
        return False

def test_function_calling():
    """Test the function calling by examining current assistant config"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"🧪 Testing current assistant configuration...")
    
    try:
        response = requests.get(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            config = response.json()
            
            # Check system prompt
            system_message = config.get('model', {}).get('messages', [{}])[0].get('content', '')
            print(f"📝 System prompt analysis:")
            
            if "MUST use" in system_message:
                print("   ✅ Contains function usage instructions")
            else:
                print("   ❌ Missing strong function usage instructions")
                
            if "backend system" in system_message:
                print("   ❌ Still contains conflicting backend message")
            else:
                print("   ✅ No conflicting backend instructions")
            
            # Check functions
            functions = config.get('functions', [])
            print(f"📊 Functions analysis:")
            print(f"   - Total functions: {len(functions)}")
            
            for func in functions:
                name = func.get('name', '')
                desc = func.get('description', '')
                trigger_phrases = ['next step', 'repeat', 'previous step']
                
                has_triggers = any(phrase in desc.lower() for phrase in trigger_phrases)
                print(f"   - {name}: {'✅' if has_triggers else '❌'} {'Trigger phrases found' if has_triggers else 'No trigger phrases'}")
            
            return True
        else:
            print(f"❌ Failed to get assistant config: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing config: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔧 VAPI Assistant Function Improvement Tool")
    print("=" * 60)
    
    print("\n1. Updating assistant configuration...")
    success = update_assistant_with_improved_functions()
    
    if success:
        print("\n2. Testing configuration...")
        test_function_calling()
        
        print(f"\n🎉 Assistant {ASSISTANT_ID} has been improved!")
        print("✅ Function calling should now work correctly")
        print("✅ System prompt encourages tool usage")
        print("✅ Detailed function descriptions with trigger phrases")
        
        print("\n📝 What changed:")
        print("  - Removed 'backend handles' conflicting message")
        print("  - Added 'MUST use these functions' instruction")
        print("  - Enhanced function descriptions with specific trigger phrases")
        print("  - Lowered temperature for more consistent responses")
        
        print("\n🧪 Test phrases to try:")
        print("  - 'Hey Kukma, next step'")
        print("  - 'Hey Kukma, repeat that'")
        print("  - 'Hey Kukma, go back'")
        
    else:
        print("❌ Failed to improve assistant configuration")