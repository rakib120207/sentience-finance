from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os
import base64

load_dotenv()

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

conversation_history = []

class Message(BaseModel):
    text: str
    emotion_hint: str = "neutral"
    voice_mode: bool = False
    image_frame: str = ""  # base64 image from camera, optional

def analyze_emotion_from_frame(image_base64: str) -> str:
    """Send camera frame to vision model, get emotion description back."""
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
Example: 'User appears frustrated, with furrowed brow and tense posture.'
If face is not visible or unclear, respond: 'Visual context unclear.'"""
                        }
                    ]
                }
            ],
            max_tokens=60
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Visual context unavailable."

@app.get("/")
def health_check():
    return {"status": "EquiVoice is running", "version": "2.0", "vision": "enabled"}

@app.post("/speak")
def speak(message: Message):
    global conversation_history

    # Analyze camera frame if provided
    visual_context = ""
    if message.image_frame:
        visual_context = analyze_emotion_from_frame(message.image_frame)

    # Build the user message with all context
    user_content = message.text
    if visual_context and visual_context != "Visual context unavailable.":
        user_content = f"[Visual context: {visual_context}] [User tone: {message.emotion_hint}]\n{message.text}"
    elif message.emotion_hint != "neutral":
        user_content = f"[User tone: {message.emotion_hint}]\n{message.text}"

    conversation_history.append({
        "role": "user",
        "content": user_content
    })

    # Keep last 10 turns
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

        return {
            "response": ai_response,
            "visual_context": visual_context,
            "status": "success",
            "turns": len(conversation_history)
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
    global conversation_history
    conversation_history = []
    return {"status": "Conversation reset"}

@app.post("/analyze-frame")
def analyze_frame_only(payload: dict):
    """Standalone endpoint to test vision without full conversation."""
    image_base64 = payload.get("image_frame", "")
    if not image_base64:
        return {"emotion": "No image provided"}
    result = analyze_emotion_from_frame(image_base64)
    return {"emotion": result}