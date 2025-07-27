#!/usr/bin/env python3

"""
Test script to verify all APIs are working correctly
"""

import os
import asyncio
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_groq_api():
    """Test Groq API for STT and LLM"""
    print("🔵 Testing Groq API...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        return False
    
    try:
        # Test LLM endpoint
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": "Hello, this is a test"}],
            "max_tokens": 50
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Groq LLM API working!")
            result = response.json()
            print(f"   Response: {result['choices'][0]['message']['content']}")
            return True
        else:
            print(f"❌ Groq API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Groq API error: {e}")
        return False

def test_openai_api():
    """Test OpenAI TTS API"""
    print("\n🔵 Testing OpenAI TTS API...")
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    try:
        # Test TTS endpoint
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": "Hello, this is a test of the text to speech system",
            "voice": "nova"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ OpenAI TTS API working!")
            print(f"   Audio length: {len(response.content)} bytes")
            return True
        else:
            print(f"❌ OpenAI API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        return False

def test_gemini_api():
    """Test Gemini API for video analysis"""
    print("\n🔵 Testing Gemini API...")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not found")
        return False
    
    try:
        # Test text generation endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        data = {
            "contents": [{
                "parts": [{
                    "text": "Hello, this is a test. Please respond with 'Gemini API working!'"
                }]
            }]
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Gemini API working!")
            result = response.json()
            if 'candidates' in result and result['candidates']:
                print(f"   Response: {result['candidates'][0]['content']['parts'][0]['text']}")
            return True
        else:
            print(f"❌ Gemini API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return False

def test_daily_api():
    """Test Daily.co API"""
    print("\n🔵 Testing Daily.co API...")
    
    api_key = os.getenv("DAILY_API_KEY")
    if not api_key:
        print("❌ DAILY_API_KEY not found")
        return False
    
    try:
        # Test room creation
        url = "https://api.daily.co/v1/rooms"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "name": "cookma-test-room",
            "properties": {
                "max_participants": 2,
                "enable_chat": False,
                "enable_screenshare": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Daily.co API working!")
            result = response.json()
            print(f"   Room URL: {result['url']}")
            
            # Clean up - delete the test room
            room_name = result['name']
            delete_url = f"https://api.daily.co/v1/rooms/{room_name}"
            requests.delete(delete_url, headers=headers)
            print("   Test room cleaned up")
            return True
        else:
            print(f"❌ Daily.co API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Daily.co API error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 Testing CookMaa Voice Assistant APIs")
    print("=" * 50)
    
    results = []
    results.append(test_groq_api())
    results.append(test_openai_api())
    results.append(test_gemini_api())
    results.append(test_daily_api())
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    
    services = ["Groq (STT/LLM)", "OpenAI (TTS)", "Gemini (Video)", "Daily.co (Rooms)"]
    for i, (service, result) in enumerate(zip(services, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {service}: {status}")
    
    all_passed = all(results)
    if all_passed:
        print("\n🎉 All APIs working! Ready to deploy voice assistant!")
        print("\n💰 Estimated monthly costs:")
        print("   - Groq (STT/LLM): ~$2/month")
        print("   - OpenAI (TTS): ~$2/month") 
        print("   - Gemini (Video): FREE")
        print("   - Daily.co (Rooms): FREE")
        print("   - Railway (Hosting): $5/month")
        print("   - TOTAL: ~$9/month")
    else:
        print("\n❌ Some APIs failed. Check your API keys and try again.")
    
    return all_passed

if __name__ == "__main__":
    main()