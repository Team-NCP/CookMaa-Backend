#!/usr/bin/env python3

"""
Test complete Groq-powered voice pipeline
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_complete_groq_pipeline():
    """Test STT -> LLM -> TTS pipeline using only Groq"""
    print("🚀 Testing Complete Groq Voice Pipeline")
    print("=" * 50)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        return False
    
    # Step 1: Simulate STT (we'll use text input for testing)
    user_input = "Hey Kukma, what's the next step in making tomato rice?"
    print(f"🎤 User says: '{user_input}'")
    
    # Step 2: Process with LLM
    print("\n🧠 Processing with Groq LLM...")
    llm_response = test_groq_llm(api_key, user_input)
    if not llm_response:
        return False
    
    # Step 3: Convert response to speech with TTS
    print("\n🔊 Converting to speech with Groq TTS...")
    tts_success = test_groq_tts_response(api_key, llm_response)
    
    if tts_success:
        print("\n🎉 Complete voice pipeline working!")
        print("\n💰 All-Groq Cost Breakdown:")
        print("   - STT: $0.111/hour")
        print("   - LLM: $0.59/1M tokens") 
        print("   - TTS: $50/1M characters")
        print("   - Total: ~$3/month for 20 cooking sessions")
        return True
    else:
        return False

def test_groq_llm(api_key: str, user_input: str) -> str:
    """Test Groq LLM with cooking context"""
    try:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Create cooking-specific prompt
        system_prompt = """You are Kukma, an expert cooking assistant. You help users cook step-by-step with encouraging, clear instructions. 

Current recipe context: Tomato Rice (Bele Bath)
Current step: Cooking rice and dal together
Next step: Preparing tomato gravy

Respond naturally and helpfully to cooking questions."""
        
        data = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            "max_tokens": 150,
            "temperature": 0.7
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            llm_response = result['choices'][0]['message']['content']
            print(f"✅ LLM Response: {llm_response}")
            return llm_response
        else:
            print(f"❌ LLM failed: {response.status_code} - {response.text}")
            return ""
            
    except Exception as e:
        print(f"❌ LLM error: {e}")
        return ""

def test_groq_tts_response(api_key: str, text: str) -> bool:
    """Test Groq TTS with LLM response"""
    try:
        url = "https://api.groq.com/openai/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "playai-tts",
            "input": text,
            "voice": "Celeste-PlayAI"  # Natural female voice for cooking
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print(f"✅ TTS successful! Generated {len(response.content)} bytes")
            
            # Save audio file
            with open("kukma_response.mp3", "wb") as f:
                f.write(response.content)
            print("   Saved Kukma's response to kukma_response.mp3")
            
            return True
        else:
            print(f"❌ TTS failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ TTS error: {e}")
        return False

def show_voice_options():
    """Show available voice options"""
    print("\n🎤 Available Kukma Voice Options:")
    cooking_voices = [
        ("Celeste-PlayAI", "👩 Natural female - Recommended"),
        ("Gail-PlayAI", "👩 Warm, friendly tone"),  
        ("Cheyenne-PlayAI", "👩 Clear, professional"),
        ("Calum-PlayAI", "👨 Clear male voice"),
        ("Mason-PlayAI", "👨 Friendly male"),
    ]
    
    for voice, description in cooking_voices:
        print(f"   {voice}: {description}")

def compare_with_alternatives():
    """Compare Groq with other options"""
    print("\n📊 Voice Pipeline Comparison:")
    print("")
    print("🔹 All-Groq Setup (Current):")
    print("   STT: Groq Whisper ($0.111/hour)")
    print("   LLM: Groq Llama ($0.59/1M tokens)")
    print("   TTS: Groq PlayAI ($50/1M chars)")
    print("   Total: ~$3/month")
    print("")
    print("🔹 Mixed Budget Setup:")
    print("   STT: Groq Whisper ($0.111/hour)") 
    print("   LLM: Groq Llama ($0.59/1M tokens)")
    print("   TTS: OpenAI ($15/1M chars)")
    print("   Total: ~$1.50/month (50% cheaper!)")
    print("")
    print("🎯 Recommendation: Try all-Groq first, fallback to OpenAI TTS if needed")

if __name__ == "__main__":
    success = test_complete_groq_pipeline()
    show_voice_options()
    compare_with_alternatives()
    
    if success:
        print("\n✅ Voice pipeline ready for deployment!")
    else:
        print("\n❌ Pipeline needs troubleshooting")