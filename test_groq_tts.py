#!/usr/bin/env python3

"""
Test Groq TTS capabilities
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

def test_groq_tts():
    """Test Groq TTS API"""
    print("🔊 Testing Groq TTS API...")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found")
        return False
    
    try:
        # Test TTS endpoint
        url = "https://api.groq.com/openai/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Test with PlayAI-TTS model and a cooking-friendly voice
        data = {
            "model": "playai-tts",
            "input": "Hello! I'm Kukma, your cooking assistant. Let's start cooking something delicious today!",
            "voice": "Celeste-PlayAI"  # Female voice good for cooking
        }
        
        print("📡 Making request to Groq TTS...")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            print("✅ Groq TTS API working!")
            print(f"   Audio length: {len(response.content)} bytes")
            
            # Save test audio
            with open("groq_tts_test.mp3", "wb") as f:
                f.write(response.content)
            print("   Saved test audio to groq_tts_test.mp3")
            
            return True
        else:
            print(f"❌ Groq TTS failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Groq TTS error: {e}")
        return False

def list_groq_voices():
    """List available Groq TTS voices"""
    print("\n🎤 Available Groq TTS Voices:")
    
    english_voices = [
        "Arista-PlayAI", "Atlas-PlayAI", "Basil-PlayAI", "Briggs-PlayAI",
        "Calum-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Chip-PlayAI",
        "Cillian-PlayAI", "Deedee-PlayAI", "Fritz-PlayAI", "Gail-PlayAI",
        "Indigo-PlayAI", "Mamaw-PlayAI", "Mason-PlayAI", "Mikail-PlayAI",
        "Mitch-PlayAI", "Quinn-PlayAI", "Thunder-PlayAI"
    ]
    
    print("📋 English Voices (19 available):")
    for i, voice in enumerate(english_voices, 1):
        gender = "👩" if voice in ["Arista-PlayAI", "Celeste-PlayAI", "Cheyenne-PlayAI", "Deedee-PlayAI", "Gail-PlayAI", "Mamaw-PlayAI"] else "👨"
        print(f"   {i:2d}. {gender} {voice}")
    
    print("\n🏆 Recommended for Cooking:")
    print("   👩 Celeste-PlayAI - Natural female voice")
    print("   👩 Gail-PlayAI - Warm, friendly tone")
    print("   👨 Calum-PlayAI - Clear male voice")

def compare_costs():
    """Compare TTS costs"""
    print("\n💰 TTS Cost Comparison:")
    print("   🔹 Groq TTS: $50/1M characters")
    print("   🔹 OpenAI TTS: $15/1M characters (95% cheaper!)")
    print("   🔹 Play.HT: $100/1M characters")
    print("")
    print("📊 Monthly Estimates (20 cooking sessions):")
    print("   - Groq: ~$2.50/month")
    print("   - OpenAI: ~$0.75/month") 
    print("   - Play.HT: ~$5/month")
    print("")
    print("🎯 Recommendation: Use OpenAI TTS for best value!")

if __name__ == "__main__":
    test_groq_tts()
    list_groq_voices()
    compare_costs()