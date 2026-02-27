from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from groq import Groq
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os, base64, json as json_module, logging
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("sentience")

load_dotenv()

# ── Firebase ──
firebase_key_b64 = os.getenv("FIREBASE_KEY_BASE64")
if firebase_key_b64:
    firebase_key_dict = json_module.loads(base64.b64decode(firebase_key_b64).decode())
    cred = credentials.Certificate(firebase_key_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase connected via env var")
else:
    if os.path.exists("firebase-key.json"):
        cred = credentials.Certificate("firebase-key.json")
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        logger.info("Firebase connected via key file")
    else:
        db = None
        logger.warning("Firebase not configured — no persistence")

# ── App ──
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Sentience Finance API", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── System Prompt ──
SENTIENCE_SYSTEM_PROMPT = """
You are Sentience, a personal financial conscience AI.
Your role: help users make emotionally-aware financial decisions.

Personality:
- Direct, warm, and honest — never preachy or judgmental
- You notice patterns others miss: the link between emotion and money
- Concise: 2-4 sentences maximum per response
- You speak like a trusted friend who understands behavioral finance

Critical rules:
- Never say "As an AI..." or "I cannot..."
- When you see [Visual: ...], the user's camera detected their emotional state — reflect it
- When you see [INTERVENTION], this is a high-stakes moment.
  Compassionately surface their past emotional spending pattern.
  Suggest a 24-hour pause. Never shame them.
- Never shame users about money. Curiosity and care, not judgment.

Tone by emotion:
- Frustrated/Angry: "I notice you're frustrated right now..."
- Sad/Lonely: "Spending when we're low sometimes feels like comfort..."
- Stressed: "Big purchases feel more urgent when we're stressed..."
- Happy: Affirm good decisions, gently flag celebratory overspending
- Neutral: Be analytical and clear
"""

# ── Constants ──
SPEND_KEYWORDS = [
    "buy", "purchase", "order", "amazon", "shop", "spend",
    "want to get", "thinking of getting", "should i get", "checkout",
    "add to cart", "grab", "treat myself", "splurge", "just bought",
    "just ordered", "just spent", "gonna buy", "thinking of buying"
]

HIGH_STRESS_EMOTIONS = [
    "frustrated", "angry", "sad", "fearful",
    "tired", "anxious", "stressed", "lonely", "confused"
]

SPEND_CATEGORIES = [
    "Food & Dining", "Shopping", "Entertainment",
    "Travel", "Health", "Technology", "Subscriptions", "Other"
]

# ── In-Memory State ──
conversation_history = []
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
session_title = None


# ─────────────────────────────────────
#  MODELS
# ─────────────────────────────────────
class Message(BaseModel):
    text: str
    emotion_hint: str = "neutral"
    image_frame: str = ""

    @validator("text")
    def text_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty")
        if len(v) > 2000:
            raise ValueError("Message too long (max 2000 chars)")
        return v

    @validator("image_frame")
    def image_not_too_large(cls, v):
        if v and len(v) > 500_000:
            raise ValueError("Image too large")
        return v


class SpendLog(BaseModel):
    amount: float
    category: str = "Other"
    description: str = ""
    emotion_hint: str = "neutral"
    visual_context: str = ""

    @validator("amount")
    def amount_valid(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return round(v, 2)

    @validator("category")
    def category_valid(cls, v):
        return v if v in SPEND_CATEGORIES else "Other"


class RegretLog(BaseModel):
    spend_id: str
    regret_score: int

    @validator("regret_score")
    def score_valid(cls, v):
        return max(0, min(10, v))


# ─────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────
def safety_check(text: str) -> str:
    crisis = ["hurt myself", "end it", "give up", "can't go on",
              "want to die", "kill myself", "no point", "disappear"]
    urgent = ["emergency", "call 911", "ambulance", "i'm in danger"]
    t = text.lower()
    if any(s in t for s in crisis):
        logger.warning("Safety: crisis signal")
        return ("I hear you — you're going through something really hard. "
                "Please reach out to someone you trust or a crisis line. You're not alone.")
    if any(s in t for s in urgent):
        logger.warning("Safety: urgent signal")
        return ("This sounds urgent — please contact emergency services immediately. "
                "Get real help first.")
    return ""


def check_intervention(text: str, emotion: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SPEND_KEYWORDS) and emotion in HIGH_STRESS_EMOTIONS


def get_emotional_spend_history(emotion: str) -> dict:
    if not db:
        return {"count": 0, "total": 0.0, "avg": 0.0, "examples": []}
    try:
        docs = (db.collection("spend_logs")
                .where("emotion_hint", "==", emotion)
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(10).get())
        data = [d.to_dict() for d in docs]
        total = sum(d.get("amount", 0) for d in data)
        count = len(data)
        examples = [f"${d['amount']:.0f} on {d.get('category','?')}" for d in data[:3]]
        return {"count": count, "total": round(total, 2),
                "avg": round(total / count, 2) if count else 0,
                "examples": examples}
    except Exception as e:
        logger.error(f"Emotional history failed: {e}")
        return {"count": 0, "total": 0.0, "avg": 0.0, "examples": []}


def analyze_emotion_from_frame(image_base64: str) -> str:
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {"type": "text",
                 "text": ("Describe this person's emotional state in ONE sentence. "
                          "Format: 'User appears [emotion], [one specific observation].' "
                          "If no face: 'Visual context unclear.'")}
            ]}],
            max_tokens=60,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Vision failed: {e}")
        return "Visual context unavailable."


def generate_session_title(first_message: str) -> str:
    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user",
                       "content": (f"3-5 word title for a financial chat starting with: "
                                   f"'{first_message[:80]}'. Return ONLY the title.")}],
            max_tokens=20,
        )
        return r.choices[0].message.content.strip()
    except Exception:
        return first_message[:30] + ("..." if len(first_message) > 30 else "")


def save_turn(user_text, ai_response, emotion, visual_context, intervention=False):
    if not db:
        return
    try:
        db.collection("sessions").document(session_id).collection("turns").add({
            "user_message": user_text,
            "ai_response": ai_response,
            "emotion_hint": emotion,
            "visual_context": visual_context,
            "intervention_triggered": intervention,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "turn_number": len(conversation_history) // 2,
        })
        db.collection("sessions").document(session_id).set({
            "title": session_title or user_text[:40],
            "total_turns": len(conversation_history) // 2,
            "last_active": firestore.SERVER_TIMESTAMP,
            "preview": user_text[:80],
            "type": "finance",
        }, merge=True)
    except Exception as e:
        logger.error(f"Firestore save failed: {e}")


# ─────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Sentience Finance running", "version": "1.0",
            "session_id": session_id, "memory": "Firestore" if db else "in-memory"}


@app.get("/health")
def health():
    return {"status": "healthy", "firebase": db is not None,
            "groq": bool(os.getenv("GROQ_API_KEY")), "version": "1.0",
            "session_id": session_id, "turns": len(conversation_history) // 2}


# ── Chat ──
@app.post("/speak")
@limiter.limit("30/minute")
def speak(request: Request, message: Message):
    global conversation_history, session_title

    logger.info(f"Turn {len(conversation_history)//2+1} | "
                f"emotion={message.emotion_hint} | vision={'yes' if message.image_frame else 'no'}")

    # Safety
    safe = safety_check(message.text)
    if safe:
        return {"response": safe, "status": "safety_intercept",
                "visual_context": "", "session_id": session_id,
                "turns": len(conversation_history) // 2}

    # Auto-title
    if not session_title and not conversation_history:
        session_title = generate_session_title(message.text)

    # Vision
    visual_context = ""
    if message.image_frame:
        visual_context = analyze_emotion_from_frame(message.image_frame)

    # Effective emotion (camera overrides chip if available)
    effective_emotion = message.emotion_hint
    if visual_context and visual_context not in ("Visual context unavailable.", "Visual context unclear."):
        vc_lower = visual_context.lower()
        for emo in HIGH_STRESS_EMOTIONS + ["happy", "calm", "neutral", "surprised"]:
            if emo in vc_lower:
                effective_emotion = emo
                break

    # Intervention
    should_intervene = check_intervention(message.text, effective_emotion)
    intervention_block = ""
    if should_intervene:
        hist = get_emotional_spend_history(effective_emotion)
        logger.info(f"INTERVENTION triggered | emotion={effective_emotion} | past_count={hist['count']}")
        if hist["count"] > 0:
            ex = ", ".join(hist["examples"]) if hist["examples"] else "similar purchases"
            intervention_block = (
                f"\n[INTERVENTION] User is {effective_emotion} and considering a purchase. "
                f"Past data: {hist['count']} purchases while {effective_emotion}, "
                f"total ${hist['total']:.0f} (avg ${hist['avg']:.0f}). "
                f"Recent: {ex}. Surface this gently. Suggest 24-hour pause."
            )
        else:
            intervention_block = (
                f"\n[INTERVENTION] User is {effective_emotion} and considering a purchase. "
                f"No prior history yet. Acknowledge their emotion and gently ask: is now the right time?"
            )

    # Build user content
    uc = message.text
    if visual_context and visual_context not in ("Visual context unavailable.", "Visual context unclear."):
        uc = f"[Visual: {visual_context}] [Tone: {effective_emotion}]{intervention_block}\n{message.text}"
    elif effective_emotion != "neutral":
        uc = f"[Tone: {effective_emotion}]{intervention_block}\n{message.text}"
    elif intervention_block:
        uc = f"{intervention_block}\n{message.text}"

    conversation_history.append({"role": "user", "content": uc})
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": SENTIENCE_SYSTEM_PROMPT}] + conversation_history,
        )
        ai_resp = r.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_resp})
        save_turn(message.text, ai_resp, effective_emotion, visual_context, should_intervene)

        return {
            "response": ai_resp,
            "visual_context": visual_context,
            "status": "intervention" if should_intervene else "success",
            "intervention": should_intervene,
            "emotion_detected": effective_emotion,
            "session_id": session_id,
            "session_title": session_title,
            "turns": len(conversation_history) // 2,
        }
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return {"response": "Having trouble — please try again.", "status": "error", "error": str(e)}


# ── Log Spend ──
@app.post("/log-spend")
@limiter.limit("60/minute")
def log_spend(request: Request, spend: SpendLog):
    if not db:
        return {"status": "error", "error": "Firebase not configured"}
    try:
        spend_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        db.collection("spend_logs").document(spend_id).set({
            "amount": spend.amount,
            "category": spend.category,
            "description": spend.description,
            "emotion_hint": spend.emotion_hint,
            "visual_context": spend.visual_context,
            "session_id": session_id,
            "regret_score": None,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "spend_id": spend_id,
        })
        logger.info(f"Spend: ${spend.amount} | {spend.category} | {spend.emotion_hint}")

        # AI acknowledgment
        try:
            cr = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": SENTIENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"User logged: ${spend.amount} on {spend.category} "
                        f"({spend.description or 'no description'}) while feeling {spend.emotion_hint}. "
                        f"Give one warm 1-2 sentence acknowledgment noting the emotional context."
                    )}
                ],
                max_tokens=80,
            )
            ai_comment = cr.choices[0].message.content
        except Exception:
            ai_comment = f"Logged ${spend.amount} on {spend.category} while {spend.emotion_hint}. I'll remember this pattern."

        return {"status": "logged", "spend_id": spend_id, "ai_comment": ai_comment}
    except Exception as e:
        logger.error(f"Log spend failed: {e}")
        return {"status": "error", "error": str(e)}


# ── Regret Score ──
@app.post("/regret")
def log_regret(regret: RegretLog):
    if not db:
        return {"status": "error", "error": "Firebase not configured"}
    try:
        db.collection("spend_logs").document(regret.spend_id).update({
            "regret_score": regret.regret_score,
            "regret_logged_at": firestore.SERVER_TIMESTAMP,
        })
        return {"status": "updated", "regret_score": regret.regret_score}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ── Insights ──
@app.get("/insights")
def get_insights():
    if not db:
        return {"chart_data": [], "category_data": [], "recent_spends": [],
                "total_logged": 0, "narrative": "Firebase not configured."}
    try:
        docs = (db.collection("spend_logs")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(200).get())

        emotion_data, category_data, total_logged, recent = {}, {}, 0, []

        for doc in docs:
            d = doc.to_dict()
            emotion = d.get("emotion_hint", "neutral")
            amount = d.get("amount", 0)
            category = d.get("category", "Other")
            regret = d.get("regret_score")
            ts = d.get("timestamp")
            total_logged += amount

            if emotion not in emotion_data:
                emotion_data[emotion] = {"total": 0, "count": 0, "regret_sum": 0, "regret_count": 0}
            emotion_data[emotion]["total"] += amount
            emotion_data[emotion]["count"] += 1
            if regret is not None:
                emotion_data[emotion]["regret_sum"] += regret
                emotion_data[emotion]["regret_count"] += 1

            category_data[category] = category_data.get(category, 0) + amount

            if len(recent) < 10:
                recent.append({
                    "amount": amount, "category": category, "emotion": emotion,
                    "description": d.get("description", ""),
                    "regret_score": regret, "spend_id": d.get("spend_id", doc.id),
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else None,
                })

        chart_data = []
        for emo, data in emotion_data.items():
            avg_regret = (round(data["regret_sum"] / data["regret_count"], 1)
                          if data["regret_count"] > 0 else None)
            chart_data.append({
                "emotion": emo, "total": round(data["total"], 2),
                "count": data["count"],
                "avg_spend": round(data["total"] / data["count"], 2),
                "avg_regret": avg_regret,
            })
        chart_data.sort(key=lambda x: x["total"], reverse=True)

        # AI narrative
        narrative = "Start logging expenses to see your emotional spending patterns."
        if chart_data:
            try:
                # Pre-format the emotional data string to avoid backslashes in the f-string
                emotion_summary = str([{x['emotion']: f"${x['total']:.0f}"} for x in chart_data[:4]])
                category_summary = str(dict(list(sorted(category_data.items(), key=lambda x: x[1], reverse=True))[:4]))

                nr = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": (
                        f"Give ONE honest 2-sentence insight about this spending: "
                        f"Total: ${total_logged:.0f}. "
                        f"By emotion: {emotion_summary}. "
                        f"By category: {category_summary}. "
                        f"Be specific and human. No fluff."
                    )}],
                    max_tokens=80,
                )
                narrative = nr.choices[0].message.content
            except Exception as e:
                logger.error(f"Narrative generation failed: {e}")
                top = chart_data[0]
                narrative = (f"You spend most when feeling {top['emotion']} "
                             f"— ${top['total']:.0f} across {top['count']} transactions.")

        return {
            "chart_data": chart_data,
            "category_data": [{"category": k, "total": round(v, 2)}
                               for k, v in sorted(category_data.items(), key=lambda x: x[1], reverse=True)],
            "recent_spends": recent,
            "total_logged": round(total_logged, 2),
            "narrative": narrative,
        }
    except Exception as e:
        logger.error(f"Insights failed: {e}")
        return {"chart_data": [], "category_data": [], "recent_spends": [],
                "total_logged": 0, "narrative": "Could not load insights.", "error": str(e)}


# ── Proactive Check-In ──
@app.get("/check-in")
@limiter.limit("10/minute")
def check_in(request: Request):
    try:
        snippet = conversation_history[-3:] if conversation_history else []
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SENTIENCE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"User quiet for 30s. Context: {snippet if snippet else 'new session'}. "
                    f"Send ONE financial check-in (under 2 sentences). "
                    f"Could be: asking how they feel about money today, "
                    f"reminding them to log a recent purchase, or sharing a quick insight."
                )},
            ],
        )
        return {"message": r.choices[0].message.content, "type": "proactive"}
    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        return {"message": "How are you feeling about your finances today?", "type": "proactive"}


# ── Session routes (preserved) ──
@app.post("/reset")
def reset():
    global conversation_history, session_id, session_title
    conversation_history = []
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_title = None
    return {"status": "reset", "new_session_id": session_id}


@app.get("/sessions")
def get_sessions():
    if not db:
        return {"sessions": [], "error": "Firebase not configured"}
    try:
        docs = (db.collection("sessions")
                .order_by("last_active", direction=firestore.Query.DESCENDING)
                .limit(30).get())
        result = []
        for s in docs:
            d = s.to_dict()
            for f in ("last_active", "started_at"):
                if f in d and hasattr(d[f], "isoformat"):
                    d[f] = d[f].isoformat()
            result.append({"id": s.id, **d})
        return {"sessions": result, "total": len(result)}
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@app.get("/sessions/{sid}/turns")
def get_turns(sid: str):
    if not db:
        return {"turns": [], "error": "Firebase not configured"}
    try:
        docs = (db.collection("sessions").document(sid)
                .collection("turns").order_by("turn_number").get())
        turns = []
        for t in docs:
            d = t.to_dict()
            if "timestamp" in d and hasattr(d["timestamp"], "isoformat"):
                d["timestamp"] = d["timestamp"].isoformat()
            turns.append(d)
        return {"turns": turns, "total": len(turns)}
    except Exception as e:
        return {"turns": [], "error": str(e)}


@app.post("/sessions/{sid}/load")
def load_session(sid: str):
    global conversation_history, session_id, session_title
    if not db:
        return {"status": "error", "error": "Firebase not configured"}
    try:
        s = db.collection("sessions").document(sid).get()
        if not s.exists:
            return {"status": "error", "error": "Session not found"}
        sd = s.to_dict()
        docs = (db.collection("sessions").document(sid)
                .collection("turns").order_by("turn_number").get())
        restored = []
        for t in docs:
            d = t.to_dict()
            um, vis, emo = d.get("user_message",""), d.get("visual_context",""), d.get("emotion_hint","neutral")
            if vis and vis not in ("Visual context unavailable.", "Visual context unclear."):
                uc = f"[Visual: {vis}] [Tone: {emo}]\n{um}"
            elif emo != "neutral":
                uc = f"[Tone: {emo}]\n{um}"
            else:
                uc = um
            restored.append({"role": "user", "content": uc})
            restored.append({"role": "assistant", "content": d.get("ai_response","")})
        conversation_history = restored[-20:]
        session_id = sid
        session_title = sd.get("title", sid)
        return {"status": "loaded", "session_id": sid,
                "session_title": session_title, "turns_loaded": len(restored)//2}
    except Exception as e:
        return {"status": "error", "error": str(e)}