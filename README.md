# Pro Forex Analyzer 📈

AI-powered forex chart analysis with institutional-grade features.

## Features
- 🤖 AI Chart Analysis (Gemini/Groq)
- 📊 744 Technical Concepts
- 🔐 User Authentication (Google OAuth)
- 📱 Responsive Design
- 🌏 Cambodia Timezone Support

## Quick Deploy to Render (FREE)

1. Fork this repo to your GitHub
2. Go to [render.com](https://render.com)
3. Click "New" → "Web Service"
4. Connect your GitHub repo
5. Add Environment Variables:
   - `GEMINI_API_KEY` - Get from https://aistudio.google.com/apikey
   - `GROQ_API_KEY` - Get from https://console.groq.com
   - `SECRET_KEY` - Any random string
   - `GOOGLE_CLIENT_ID` - From Google Cloud Console
   - `GOOGLE_CLIENT_SECRET` - From Google Cloud Console

6. Click "Create Web Service"

## Local Development

```bash
cd forex-analyzer-pro
pip install -r requirements.txt
python app.py
```

Visit http://localhost:5000

## Environment Variables

Copy `.env.example` to `.env` and fill in your API keys.

## License
MIT
