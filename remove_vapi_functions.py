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

def completely_remove_functions():
    """Completely remove deprecated functions from assistant"""
    
    headers = {
        "Authorization": f"Bearer {VAPI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    print(f"🗑️ Removing all deprecated functions from assistant...")
    
    # Try to explicitly remove functions by setting to null/empty
    minimal_update = {
        "functions": None,  # Try to explicitly remove
    }
    
    try:
        # First attempt: set functions to null
        response = requests.patch(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            json=minimal_update,
            timeout=30
        )
        
        print(f"Response: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            print("✅ Functions set to null")
        else:
            print(f"❌ Failed to set functions to null")
            
            # Second attempt: set functions to empty array
            print("🔄 Trying empty array...")
            minimal_update = {"functions": []}
            response = requests.patch(
                f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
                headers=headers,
                json=minimal_update,
                timeout=30
            )
            
            print(f"Response: {response.status_code} - {response.text}")
            
        # Verify current state
        print("\n🔍 Checking current configuration...")
        response = requests.get(
            f"https://api.vapi.ai/assistant/{ASSISTANT_ID}",
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            config = response.json()
            functions = config.get("functions", [])
            tools = config.get("model", {}).get("tools", [])
            
            print(f"📊 Current state:")
            print(f"   Functions (deprecated): {len(functions)} items")
            print(f"   Model Tools (new): {len(tools)} items")
            
            if len(functions) == 0:
                print("✅ Functions successfully removed!")
                return True
            else:
                print(f"❌ Functions still present: {functions}")
                return False
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🗑️ VAPI Functions Removal Tool")
    print("=" * 50)
    
    success = completely_remove_functions()
    
    if success:
        print(f"\n🎉 Successfully removed deprecated functions!")
        print("✅ Assistant now uses only Tools (dashboard configuration)")
        print("✅ Function calls should now reach webhook properly")
    else:
        print("\n❌ Could not remove deprecated functions")
        print("💡 Suggestion: Try manually removing functions via VAPI dashboard")