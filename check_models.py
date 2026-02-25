from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
models = client.models.list()

print("=== Vision & Multimodal Models ===")
for m in models.data:
    if any(x in m.id.lower() for x in ['vision', 'llama-3.2', 'llava', 'scout', 'llama-4']):
        print(m.id)

print("\n=== All Available Models ===")
for m in models.data:
    print(m.id)