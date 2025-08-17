#!/usr/bin/env python3

import requests
import json

def test_session_linking():
    """Test if the webhook can find and link to existing cooking sessions"""
    
    backend_url = "https://cookmaa-backend-production.up.railway.app"
    
    # First, check what sessions exist
    print("🔍 Checking existing sessions...")
    response = requests.get(f"{backend_url}/sessions")
    if response.status_code == 200:
        sessions_data = response.json()
        sessions = sessions_data.get('active_sessions', [])
        print(f"✅ Found {len(sessions)} active sessions")
        
        if sessions:
            latest_session = sessions[0]  # Assuming first is most recent
            print(f"🎯 Latest session: {latest_session['recipe_title']} - Step {latest_session['current_step']}")
        else:
            print("❌ No active sessions found")
            return
    else:
        print(f"❌ Failed to get sessions: {response.status_code}")
        return
    
    # Now simulate a VAPI webhook call with a transcript
    print("\n🧪 Testing webhook with simulated VAPI call...")
    
    test_payload = {
        "message": {
            "type": "transcript",
            "transcript": "hey kukma, what's the next step?",
            "role": "user"
        },
        "call": {
            "id": "test-session-linking-123"
        }
    }
    
    try:
        response = requests.post(
            f"{backend_url}/webhook/vapi-message",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"📡 Webhook response status: {response.status_code}")
        
        if response.status_code == 200:
            try:
                response_data = response.json() if response.content else response.text
                print(f"🤖 Assistant response: {response_data}")
                
                # Check if it found the recipe context
                if isinstance(response_data, str):
                    if "Bisi Bele Bath" in response_data or "step" in response_data.lower():
                        print("✅ SUCCESS: Assistant found recipe context!")
                    else:
                        print("❌ FAILED: Assistant doesn't have recipe context")
                        print(f"   Response: {response_data}")
                        
            except json.JSONDecodeError:
                print(f"📝 Raw response: {response.text}")
        else:
            print(f"❌ Webhook failed: {response.text}")
            
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_session_linking()