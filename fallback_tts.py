#!/usr/bin/env python3

"""
Fallback TTS options for budget voice assistant
"""

import os
import requests
from typing import Optional

class GroqTTSFallback:
    """Use Groq for both LLM and TTS synthesis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.groq.com/openai/v1"
    
    def synthesize_speech(self, text: str, voice: str = "nova") -> Optional[bytes]:
        """Synthesize speech using Groq TTS (if available)"""
        # Note: This is a placeholder - Groq may not have TTS yet
        # But we can use their text generation for voice-like responses
        return None

class PlayHTTTS:
    """Play.HT TTS service - budget alternative"""
    
    def __init__(self, api_key: str, user_id: str):
        self.api_key = api_key
        self.user_id = user_id
        self.base_url = "https://api.play.ht/api/v2"
    
    def synthesize_speech(self, text: str, voice: str = "s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json") -> Optional[bytes]:
        """Synthesize speech using Play.HT"""
        try:
            headers = {
                "AUTHORIZATION": f"Bearer {self.api_key}",
                "X-USER-ID": self.user_id,
                "Content-Type": "application/json"
            }
            
            data = {
                "text": text,
                "voice": voice,
                "quality": "medium",
                "output_format": "mp3",
                "speed": 0.9
            }
            
            response = requests.post(f"{self.base_url}/tts", headers=headers, json=data)
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"Play.HT TTS failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Play.HT TTS error: {e}")
            return None

class BudgetTTSManager:
    """Manages multiple TTS fallbacks"""
    
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.groq_key = os.getenv("GROQ_API_KEY")
        self.playht_key = os.getenv("PLAYHT_API_KEY")
        self.playht_user = os.getenv("PLAYHT_USER_ID")
    
    def get_available_tts(self) -> str:
        """Get the best available TTS service"""
        
        # Try OpenAI first (best quality)
        if self.openai_key and self.test_openai_quota():
            return "openai"
        
        # Try Play.HT (good budget option)
        if self.playht_key and self.playht_user:
            return "playht"
        
        # Fallback to system TTS
        return "system"
    
    def test_openai_quota(self) -> bool:
        """Test if OpenAI has available quota"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "tts-1",
                "input": "test",
                "voice": "nova"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=data
            )
            
            return response.status_code == 200
            
        except:
            return False
    
    def synthesize(self, text: str, voice: str = "nova") -> Optional[bytes]:
        """Synthesize speech using best available service"""
        service = self.get_available_tts()
        
        if service == "openai":
            return self._openai_tts(text, voice)
        elif service == "playht":
            tts = PlayHTTTS(self.playht_key, self.playht_user)
            return tts.synthesize_speech(text)
        else:
            print("Using system TTS fallback")
            return None
    
    def _openai_tts(self, text: str, voice: str) -> Optional[bytes]:
        """OpenAI TTS implementation"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": "tts-1",
                "input": text,
                "voice": voice
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/speech",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.content
            else:
                print(f"OpenAI TTS failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"OpenAI TTS error: {e}")
            return None

def test_budget_tts():
    """Test budget TTS options"""
    print("🔊 Testing Budget TTS Options...")
    
    manager = BudgetTTSManager()
    available_service = manager.get_available_tts()
    
    print(f"Best available TTS: {available_service}")
    
    test_text = "Hello! This is Kukma, your cooking assistant. Let's start cooking!"
    audio_data = manager.synthesize(test_text)
    
    if audio_data:
        print(f"✅ TTS working! Generated {len(audio_data)} bytes of audio")
        
        # Save test audio file
        with open("test_tts.mp3", "wb") as f:
            f.write(audio_data)
        print("   Saved test audio to test_tts.mp3")
    else:
        print("❌ No TTS service available")

if __name__ == "__main__":
    test_budget_tts()