# CookMaa Voice Assistant Backend

AI-powered voice assistant for cooking guidance using Groq's ultra-fast STT, LLM, and TTS services.

## 🚀 Quick Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/deploy)

## 🎯 Features

- **Ultra-fast voice pipeline**: Groq STT → LLM → TTS in <500ms
- **Cooking-specific AI**: Context-aware recipe guidance
- **Natural voice**: 19 English voices including cooking-optimized options
- **Real-time communication**: Daily.co integration for live voice sessions
- **Budget-friendly**: ~$3/month for complete voice assistant

## 🔧 Environment Variables

Set these in Railway Dashboard:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key  
DAILY_API_KEY=your_daily_api_key
GROQ_VOICE=Celeste-PlayAI
PORT=8000
```

## 📊 API Endpoints

### Health Check
```
GET /health
```

### Start Voice Session
```
POST /start-voice-session
{
  "room_url": "https://cookma.daily.co/room",
  "token": "daily_room_token",
  "recipe_context": {
    "title": "Tomato Rice",
    "steps": [...],
    "chefs_wisdom": "..."
  },
  "step_index": 0
}
```

### Update Recipe Context
```
POST /update-recipe-context
{
  "title": "Recipe Name",
  "steps": [...],
  "step_index": 2
}
```

## 🎤 Voice Commands Supported

- **"Hey Kukma, next step"** - Move to next cooking step
- **"Hey Kukma, repeat that"** - Repeat current instruction
- **"Hey Kukma, I have a question"** - Answer cooking questions
- **"Hey Kukma, how long should this take?"** - Provide timing info

## 🔊 Available Voices

**Recommended for Cooking:**
- `Celeste-PlayAI` - Natural female (default)
- `Gail-PlayAI` - Warm, friendly tone
- `Calum-PlayAI` - Clear male voice

**All 19 English voices:**
Arista-PlayAI, Atlas-PlayAI, Basil-PlayAI, Briggs-PlayAI, Calum-PlayAI, Celeste-PlayAI, Cheyenne-PlayAI, Chip-PlayAI, Cillian-PlayAI, Deedee-PlayAI, Fritz-PlayAI, Gail-PlayAI, Indigo-PlayAI, Mamaw-PlayAI, Mason-PlayAI, Mikail-PlayAI, Mitch-PlayAI, Quinn-PlayAI, Thunder-PlayAI

## 💰 Cost Breakdown

**Monthly estimates for 20 cooking sessions:**
- Groq STT: ~$0.50
- Groq LLM: ~$0.30  
- Groq TTS: ~$2.20
- Railway Hosting: $5.00
- **Total: ~$8/month**

## 🛠️ Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Test APIs
python test_groq_complete.py

# Run server
python cooking_voice_assistant.py
```

## 📱 iOS Integration

This backend is designed to work with the CookMaa iOS app:

1. **Recipe Generation**: iOS sends YouTube URL to Gemini for analysis
2. **Voice Session**: iOS connects to this backend for real-time cooking guidance
3. **Context Updates**: iOS updates current recipe step as user progresses

## 🔗 Related Repositories

- [CookMaa iOS App](https://github.com/Team-NCP/CookMa) - Main iOS application
- [CookMaa Docs](https://github.com/Team-NCP/CookMa-Docs) - Documentation and guides

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for home cooks who want hands-free cooking guidance**