from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are EquiVoice, a real-time assistive communication AI. You speak with warmth, clarity, and composure. You help people express themselves confidently."
        },
        {
            "role": "user", 
            "content": "Introduce yourself briefly."
        }
    ]
)

print(response.choices[0].message.content)