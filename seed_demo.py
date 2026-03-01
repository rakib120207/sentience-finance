"""
seed_demo.py — Sentience Finance Demo Seeder
=============================================
Run this ONCE before your hackathon demo to pre-populate Firestore
with realistic stressed-spending data. This ensures:

  ✓ Vulnerability meter shows 8.2/10 immediately (not 0)
  ✓ Intervention modal shows REAL numbers (not "no data yet")
  ✓ Psychological Ledger has a populated Danger Emotion card
  ✓ Pattern report endpoint returns real data

Usage:
    python seed_demo.py

Requirements:
    pip install firebase-admin python-dotenv

The data matches the DEMO_SAFETY_DATA in index.html exactly.
Run once per fresh Firestore environment. Safe to re-run (adds more data).
"""

import os, base64, json, uuid
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

import firebase_admin
from firebase_admin import credentials, firestore

# ── Firebase init ──────────────────────────────────────────────────
firebase_key_b64 = os.getenv("FIREBASE_KEY_BASE64")
if firebase_key_b64:
    key_dict = json.loads(base64.b64decode(firebase_key_b64).decode())
    cred = credentials.Certificate(key_dict)
elif os.path.exists("firebase-key.json"):
    cred = credentials.Certificate("firebase-key.json")
else:
    raise RuntimeError("No Firebase credentials found. Set FIREBASE_KEY_BASE64 or place firebase-key.json here.")

firebase_admin.initialize_app(cred)
db = firestore.client()

# ── Demo spending data ─────────────────────────────────────────────
# These purchases will show in the Psychological Ledger and be
# referenced verbatim during the intervention: "5 purchases, $640 total,
# avg $128 each, 7.8/10 regret."

STRESSED_PURCHASES = [
    {"amount": 189.99, "category": "Technology",   "description": "Mechanical keyboard",  "regret": 9},
    {"amount": 134.00, "category": "Shopping",     "description": "Sneakers (impulse)",   "regret": 8},
    {"amount": 94.99,  "category": "Entertainment","description": "Gaming subscription",  "regret": 7},
    {"amount": 119.00, "category": "Technology",   "description": "Smart watch band",     "regret": 8},
    {"amount": 102.00, "category": "Shopping",     "description": "Jacket I never wore",  "regret": 8},
]
# Total = $639.98 ≈ $640 | avg = $128 | avg regret = 8.0

HAPPY_PURCHASES = [
    {"amount": 5.50,  "category": "Food & Dining", "description": "Coffee treat",   "regret": 1},
    {"amount": 24.99, "category": "Entertainment", "description": "Movie night",     "regret": 2},
    {"amount": 14.00, "category": "Food & Dining", "description": "Birthday lunch",  "regret": 0},
]
# Happy state: low regret — shows contrast in the ledger

NEUTRAL_PURCHASES = [
    {"amount": 67.50, "category": "Health",        "description": "Gym supplement",  "regret": 3},
    {"amount": 42.00, "category": "Subscriptions", "description": "Annual app sub",  "regret": 2},
]

# ── Write to Firestore ─────────────────────────────────────────────
def seed_purchases(purchases, emotion, days_ago_start=30):
    count = 0
    for i, p in enumerate(purchases):
        spend_id = f"demo_{emotion}_{i}_{uuid.uuid4().hex[:8]}"
        # Spread purchases over the past month (more realistic)
        days_offset = days_ago_start - (i * 4)
        ts = datetime.utcnow() - timedelta(days=days_offset, hours=(i * 3))

        doc_data = {
            "amount":       round(p["amount"], 2),
            "category":     p["category"],
            "description":  p["description"],
            "emotion_hint": emotion,
            "visual_context": "",
            "session_id":   "demo_seed",
            "regret_score": p.get("regret"),
            "regret_logged_at": ts + timedelta(days=1),
            "timestamp":    ts,
            "spend_id":     spend_id,
        }
        db.collection("spend_logs").document(spend_id).set(doc_data)
        count += 1
        print(f"  ✓ [{emotion:10s}] ${p['amount']:7.2f} — {p['description'][:30]}"
              f"  regret={p.get('regret')}/10")
    return count

print("\n🌱 Sentience Demo Seeder")
print("=" * 50)

print("\n📍 Seeding STRESSED purchases (high regret — this is your demo data):")
n = seed_purchases(STRESSED_PURCHASES, "stressed")

print(f"\n📍 Seeding HAPPY purchases (low regret — shows healthy contrast):")
n += seed_purchases(HAPPY_PURCHASES, "happy")

print(f"\n📍 Seeding NEUTRAL purchases:")
n += seed_purchases(NEUTRAL_PURCHASES, "neutral")

print(f"\n✅ Done. {n} demo transactions written to Firestore.")
print("\nExpected demo results:")
print("  • Vulnerability (stressed): ~8.2/10")
print("  • Danger Emotion card: 😰 Stressed")
print("  • Intervention text: '5 purchases, $640 total, avg $128 each, 7.8/10 regret'")
print("\nRun your backend and open the app. The Ledger and meter will be populated.")
print("The Intervention modal Safety Net (Ctrl+Shift+S) uses these exact numbers.\n")