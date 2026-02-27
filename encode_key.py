import base64
import json

with open("firebase-key.json", "r") as f:
    key_data = json.load(f)

encoded = base64.b64encode(json.dumps(key_data).encode()).decode()
print(encoded)