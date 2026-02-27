# Sentience Finance
### *The AI that understands why you spend, not just what you spend.*

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)](https://fastapi.tiangolo.com)
[![Firebase](https://img.shields.io/badge/Firebase-Firestore-orange)](https://firebase.google.com)

---

## What It Does

Sentience is a **multimodal financial conscience AI**. It detects your emotional state via camera or manual tone selection, tracks what you spend and *how you felt* when you spent it, and — critically — **intervenes in real time** before impulsive purchases happen when you're emotionally vulnerable.

**The demo moment:** User looks stressed → types "I'm thinking of buying a new iPhone" → Sentience shows their past 7 stressed purchases totaling $840 → asks them to wait 24 hours.

---

## Features

| Feature | What it does |
|---|---|
| **Emotional Spending Map** | Bar chart of spending by emotional state — the core visual |
| **Real-time Intervention** | Detects spend intent + stress → surfaces your own history |
| **Camera Emotion Detection** | Llama 4 Scout vision model reads facial expression |
| **Manual Tone Chips** | Fallback when camera is off |
| **Log Spend** | Record amount, category, and emotional state per transaction |
| **Regret Scoring** | Rate past purchases 0–10 to build behavioral data over time |
| **Proactive Check-ins** | AI initiates after 30s inactivity |
| **Persistent Memory** | All sessions stored in Firestore, reloadable |
| **Session History** | Full chat history sidebar, load any past session |

---

## Tech Stack

```
Frontend    HTML · CSS · Vanilla JS (zero frameworks, zero external fonts)
Backend     FastAPI · Python 3.11
AI          Groq (llama-3.3-70b + llama-4-scout vision)
Database    Firebase Firestore
Hosting     Render.com (backend) · GitHub Pages (frontend)
```

---

## Local Setup

### 1. Clone
```bash
git clone https://github.com/YOUR_USERNAME/sentience-finance.git
cd sentience-finance
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment variables
Create a `.env` file in the root (never commit this):
```env
GROQ_API_KEY=your_groq_key_here
FIREBASE_KEY_BASE64=your_base64_encoded_firebase_key
```

**How to get FIREBASE_KEY_BASE64:**
```bash
# On Mac/Linux:
base64 -i firebase-key.json | tr -d '\n'

# On Windows PowerShell:
[Convert]::ToBase64String([IO.File]::ReadAllBytes("firebase-key.json"))
```
Paste the output as the value of `FIREBASE_KEY_BASE64`.

### 4. Run locally
```bash
uvicorn main:app --reload
# Backend running at http://localhost:8000

# Open index.html directly in Chrome
# (Voice input requires Chrome — Firefox blocks Web Speech API)
```

### 5. Verify
Visit `http://localhost:8000/health` — should return:
```json
{"status": "healthy", "firebase": true, "groq": true}
```

---

## Deployment

### Backend → Render.com (Free)

1. Push repo to GitHub (make sure `.env` and `firebase-key.json` are in `.gitignore`)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — settings are pre-filled
5. Add environment variables in Render dashboard:
   - `GROQ_API_KEY` → your key
   - `FIREBASE_KEY_BASE64` → your base64 string
6. Click **Deploy** → wait ~3 minutes
7. Copy your Render URL (e.g. `https://sentience-finance-api.onrender.com`)

### Frontend → GitHub Pages

1. In `index.html`, update line ~689:
   ```js
   : 'https://sentience-finance-api.onrender.com'  // ← your Render URL
   ```
2. Push to GitHub
3. Go to repo Settings → Pages → Source: `main` branch → `/ (root)`
4. Your frontend URL: `https://YOUR_USERNAME.github.io/sentience-finance/`

### ⚡ Update CORS (important)

After deploying frontend, update `main.py` to allow your GitHub Pages domain:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://YOUR_USERNAME.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:5500",  # VS Code Live Server
    ],
    ...
)
```
Then redeploy backend on Render.

---

## Backend → Google Cloud Run (When Credits Available)

```bash
# One-time setup
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com

# Store secrets
echo -n "your_groq_key" | gcloud secrets create GROQ_API_KEY --data-file=-
echo -n "your_firebase_b64" | gcloud secrets create FIREBASE_KEY_BASE64 --data-file=-

# Deploy
gcloud run deploy sentience-finance-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-secrets="GROQ_API_KEY=GROQ_API_KEY:latest,FIREBASE_KEY_BASE64=FIREBASE_KEY_BASE64:latest"

# Get URL
gcloud run services describe sentience-finance-api --region us-central1 --format='value(status.url)'
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Backend status + Firebase check |
| `/speak` | POST | Main chat with intervention detection |
| `/log-spend` | POST | Record a transaction with emotion |
| `/regret` | POST | Score a past spend 0–10 |
| `/insights` | GET | Emotional spending map data |
| `/check-in` | GET | Proactive AI message |
| `/reset` | POST | New session |
| `/sessions` | GET | All past sessions |
| `/sessions/{id}/turns` | GET | Turns for a session |
| `/sessions/{id}/load` | POST | Restore session into memory |

---

## Firestore Schema

```
spend_logs/
  {spend_id}/
    amount: float
    category: string
    description: string
    emotion_hint: string       ← the key field
    visual_context: string
    regret_score: int | null   ← filled 24h later
    timestamp: Timestamp

sessions/
  {session_id}/
    title: string
    total_turns: int
    last_active: Timestamp
    type: "finance"
    turns/
      {turn_id}/
        user_message: string
        ai_response: string
        emotion_hint: string
        visual_context: string
        intervention_triggered: bool
```

---

## Lighthouse Scores

| Metric | Score |
|---|---|
| Performance | 96+ |
| Accessibility | 99 |
| Best Practices | 99 |
| SEO | 96 |

---

## Roadmap

- [x] Real-time emotion detection via camera
- [x] Financial intervention system
- [x] Psychological Ledger (emotion × spend chart)
- [x] Regret scoring
- [x] Persistent session memory
- [ ] Gemini Live API (real-time audio streaming)
- [ ] Predictive behavior modeling (ChatGPT phase)
- [ ] Confidence scores on predictions
- [ ] CSV/bank import
- [ ] Mobile app (PWA)

---

## License

MIT