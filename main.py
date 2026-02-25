from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize app and AI client
app = FastAPI(title="EquiVoice API")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# EquiVoice personality - this is your system prompt
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
- If input is unclear, ask ONE simple clarifying question
- Always preserve the dignity of the person you're helping
"""

# Conversation memory - stores last 10 messages
conversation_history = []

# Define what a message looks like
class Message(BaseModel):
    text: str
    emotion_hint: str = "neutral"
    voice_mode: bool = False  # optional, from camera later

# Health check endpoint - proves server is running
@app.get("/")
def health_check():
    return {"status": "EquiVoice is running", "version": "1.0"}

# Main chat endpoint
@app.post("/speak")
def speak(message: Message):
    global conversation_history
    
    # Add user message to history
    conversation_history.append({
        "role": "user",
        "content": f"[Emotion detected: {message.emotion_hint}] {message.text}"
    })
    
    # Keep only last 10 exchanges (memory management)
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]
    
    try:
        # Send to AI with full conversation history
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": EQUIVOICE_SYSTEM_PROMPT}
            ] + conversation_history
        )
        
        ai_response = response.choices[0].message.content
        
        # Add AI response to history
        conversation_history.append({
            "role": "assistant",
            "content": ai_response
        })
        
        return {
            "response": ai_response,
            "status": "success",
            "turns": len(conversation_history)
        }
        
    except Exception as e:
        return {
            "response": "I'm having a moment of difficulty. Could you try again?",
            "status": "error",
            "error": str(e)
        }

# Clear conversation memory
@app.post("/reset")
def reset_conversation():
    global conversation_history
    conversation_history = []
    return {"status": "Conversation reset", "message": "Ready for new session"}