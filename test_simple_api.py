#!/usr/bin/env python3

"""
Test the simplified voice assistant API
"""

import requests
import json
import time

def test_simple_api():
    """Test the simplified FastAPI endpoints"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Simplified CookMaa Voice Assistant API")
    print("=" * 60)
    
    # Test 1: Health check
    print("1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            health = response.json()
            print("✅ Health check passed")
            print(f"   Status: {health['status']}")
            print(f"   APIs: {health['apis']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure server is running:")
        print("   python3 simple_voice_assistant.py")
        return False
    
    # Test 2: Voice test
    print("\n2️⃣ Testing voice synthesis...")
    try:
        response = requests.get(f"{base_url}/test-voice")
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice test passed")
            print(f"   Text: {result['text']}")
            print(f"   Audio generated: {result['audio_generated']}")
            print(f"   Audio size: {result['audio_size']} bytes")
        else:
            print(f"❌ Voice test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Voice test error: {e}")
    
    # Test 3: Process voice command
    print("\n3️⃣ Testing voice command processing...")
    try:
        command_data = {
            "transcript": "Hey Kukma, what's the next step?",
            "recipe_context": {
                "title": "Tomato Rice",
                "cuisine": "Indian",
                "servings": 2,
                "difficulty": "Easy",
                "total_time_formatted": "45 min",
                "steps": [
                    {"instruction": "Wash and soak rice for 30 minutes"},
                    {"instruction": "Heat oil in a pan and add spices"},
                    {"instruction": "Add tomatoes and cook until soft"}
                ],
                "chefs_wisdom": "Use ripe tomatoes for best flavor"
            },
            "step_index": 0
        }
        
        response = requests.post(
            f"{base_url}/process-voice-command",
            json=command_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice command processing passed")
            print(f"   User said: {result['transcript']}")
            print(f"   Kukma responded: {result['response_text']}")
            print(f"   Audio generated: {result['has_audio']}")
            print(f"   Current step: {result['current_step']}")
        else:
            print(f"❌ Voice command failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Voice command error: {e}")
    
    # Test 4: Start voice session
    print("\n4️⃣ Testing voice session creation...")
    try:
        session_data = {
            "recipe_context": {
                "title": "Tomato Rice",
                "cuisine": "Indian",
                "servings": 2,
                "steps": [{"instruction": "Cook rice"}]
            },
            "step_index": 0
        }
        
        response = requests.post(
            f"{base_url}/start-voice-session",
            json=session_data,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Voice session creation passed")
            print(f"   Room URL: {result['room_url']}")
            print(f"   Bot name: {result['bot_name']}")
        else:
            print(f"❌ Voice session failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Voice session error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 API testing completed!")
    print("\n💡 To run the server manually:")
    print("   cd /Users/sudhanvaacharya/Desktop/Code\ Projects/CookMaa/backend")
    print("   python3 simple_voice_assistant.py")
    print("\n🌐 API will be available at: http://localhost:8000")
    print("📖 Interactive docs at: http://localhost:8000/docs")

if __name__ == "__main__":
    test_simple_api()