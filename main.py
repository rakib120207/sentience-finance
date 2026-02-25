from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

load_dotenv()

# ── Firebase Init ──
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ── App Init ──
app = FastAPI(title="EquiVoice API")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EQUIVOICE_SYSTEM_PROMPT = """
You are EquiVoice, a real-time assistive communication AI designed
to help people with speech impairments express themselves.

Your personality:
- Warm, calm, empowering, and patient
- Concise: 1-3 sentences maximum per response
- Never clinical or robotic — always human
- When someone is frustrated: become calmer and more supportive
- When someone is confused: simplify and clarify gently
- When someone is urgent: respond quickly and directly

Critical rules:
- Never say "As an AI..." or "I'm just a language model..."
- Never refuse to help someone express themselves
- If visual emotion context is provided, subtly adapt your tone
- Always preserve the dignity of the person you're helping
"""

# In-memory conversation history (last 10 turns)
conversation_history = []
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

class Message(BaseModel):
    text: str
    emotion_hint: str = "neutral"
    voice_mode: bool = False
    image_frame: str = ""


def analyze_emotion_from_frame(image_base64: str) -> str:
    """Send camera frame to vision model, get emotion back."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Look at this person's face and body language.
In ONE short sentence, describe their apparent emotional state.
Focus only on: facial expression, posture, eye contact.
Format: 'User appears [emotion], [one observation].'
If face not visible: 'Visual context unclear.'"""
                        }
                    ]
                }
            ],
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return "Visual context unavailable."


def save_turn_to_firestore(user_text, ai_response, emotion, visual_context):
    """Save each conversation turn to Firestore."""
    try:
        db.collection("sessions").document(session_id).collection("turns").add({
            "user_message": user_text,
            "ai_response": ai_response,
            "emotion_hint": emotion,
            "visual_context": visual_context,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "turn_number": len(conversation_history) // 2
        })
        # Also update session metadata
        db.collection("sessions").document(session_id).set({
            "started_at": firestore.SERVER_TIMESTAMP,
            "total_turns": len(conversation_history) // 2,
            "last_active": firestore.SERVER_TIMESTAMP
        }, merge=True)
    except Exception as e:
        print(f"Firestore write failed (non-critical): {e}")


@app.get("/")
def health_check():
    return {
        "status": "EquiVoice is running",
        "version": "3.0",
        "vision": "enabled",
        "memory": "Firestore connected",
        "session_id": session_id
    }


@app.post("/speak")
def speak(message: Message):
    global conversation_history

    # Analyze camera frame if provided
    visual_context = ""
    if message.image_frame:
        visual_context = analyze_emotion_from_frame(message.image_frame)

    # Build user message with context
    user_content = message.text
    if visual_context and visual_context != "Visual context unavailable.":
        user_content = f"[Visual: {visual_context}] [Tone: {message.emotion_hint}]\n{message.text}"
    elif message.emotion_hint != "neutral":
        user_content = f"[Tone: {message.emotion_hint}]\n{message.text}"

    conversation_history.append({
        "role": "user",
        "content": user_content
    })

    # Keep last 10 turns in memory
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EQUIVOICE_SYSTEM_PROMPT}
            ] + conversation_history
        )

        ai_response = response.choices[0].message.content

        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })

        # Save to Firestore (non-blocking — won't crash if it fails)
        save_turn_to_firestore(
            message.text,
            ai_response,
            message.emotion_hint,
            visual_context
        )

        return {
            "response": ai_response,
            "visual_context": visual_context,
            "status": "success",
            "session_id": session_id,
            "turns": len(conversation_history) // 2
        }

    except Exception as e:
        return {
            "response": "I'm having a moment of difficulty. Could you try again?",
            "visual_context": "",
            "status": "error",
            "error": str(e)
        }


@app.post("/reset")
def reset_conversation():
    global conversation_history, session_id
    conversation_history = []
    # New session ID on reset
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "status": "Conversation reset",
        "new_session_id": session_id
    }


@app.get("/sessions")
def get_sessions():
    """Returns all saved sessions — shows judges the memory system."""
    try:
        sessions = db.collection("sessions").order_by(
            "last_active",
            direction=firestore.Query.DESCENDING
        ).limit(10).get()
        return {
            "sessions": [
                {"id": s.id, **s.to_dict()} for s in sessions
            ]
        }
    except Exception as e:
        return {"sessions": [], "error": str(e)}