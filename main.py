from fastapi import FastAPI, Request, Header, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from groq import Groq
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os, base64, json as json_module, logging, uuid, asyncio
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Optional

# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s"
)
logger = logging.getLogger("sentience")

load_dotenv()

# ══════════════════════════════════════════════════════════════════
#  FIREBASE
# ══════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════
#  AI CLIENT
#  ─────────────────────────────────────────────────────────────────
#  GEMINI MIGRATION NOTE:
#  When switching to Gemini Live, replace this client and the two
#  helper functions `_chat()` and `_vision()` below.
#  All routes call ONLY those two helpers — nothing else touches
#  the model directly. Swap the helpers, zero other changes needed.
# ══════════════════════════════════════════════════════════════════
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

TEXT_MODEL  = "llama-3.3-70b-versatile"
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

def _chat(messages: list, max_tokens: int = 300) -> str:
    """
    Central text inference call.
    GEMINI SWAP: replace body with google.generativeai call.
    """
    r = groq_client.chat.completions.create(
        model=TEXT_MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()


def _vision(image_base64: str, prompt: str) -> str:
    """
    Central vision inference call.
    GEMINI SWAP: replace body with Gemini Vision / Gemini Live call.
    """
    r = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": prompt}
        ]}],
        max_tokens=60,
    )
    return r.choices[0].message.content.strip()


# ══════════════════════════════════════════════════════════════════
#  APP
# ══════════════════════════════════════════════════════════════════
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Sentience Finance API", version="2.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://rakib120207.github.io",
        "http://localhost:8000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "null",  # file:// opens as "null" origin
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════
#  PER-SESSION IN-MEMORY STATE
#  ─────────────────────────────────────────────────────────────────
#  CRITICAL FIX: replaces dangerous global conversation_history.
#  Each browser tab / user gets its own isolated state keyed by
#  a UUID session_id that the frontend generates on load and sends
#  with every request via the X-Session-ID header.
#  Capped at MAX_SESSIONS to prevent memory leak on Render free tier.
# ══════════════════════════════════════════════════════════════════
MAX_SESSIONS = 200
_sessions: dict[str, dict] = {}


def get_session(sid: str) -> dict:
    if sid not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            # Evict the oldest session (simple FIFO)
            oldest = next(iter(_sessions))
            del _sessions[oldest]
            logger.info(f"Evicted oldest session {oldest}")
        _sessions[sid] = {
            "conversation_history": [],
            "session_title": None,
            "created_at": datetime.utcnow().isoformat(),
        }
        logger.info(f"New session created: {sid}")
    return _sessions[sid]


# ══════════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════
SENTIENCE_SYSTEM_PROMPT = """
You are Sentience, a personal financial conscience AI.
Your role: help users make emotionally-aware financial decisions.

Personality:
- Direct, warm, and honest — never preachy or judgmental
- You notice patterns others miss: the link between emotion and money
- Concise: 2-4 sentences maximum per response
- You speak like a trusted friend who deeply understands behavioral finance

Critical rules:
- Never say "As an AI..." or "I cannot..."
- When you see [Visual: ...], the user's camera detected their emotional state — reflect it naturally
- When you see [INTERVENTION], this is the most important moment.
  Start with "Hold on." Then compassionately surface their EXACT past pattern with numbers.
  Suggest a 24-hour pause. End with one question. Never shame them.
- Never shame users about money. Curiosity and care, not judgment.

Tone by emotion:
- Frustrated/Angry: "I notice you're frustrated right now..."
- Sad/Lonely: "Spending when we're low sometimes feels like comfort..."
- Stressed: "Big purchases feel more urgent when we're stressed..."
- Happy: Affirm good decisions, gently flag celebratory overspending
- Neutral: Be analytical and clear
"""

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════
SPEND_KEYWORDS = [
    "buy", "purchase", "order", "amazon", "shop", "spend",
    "want to get", "thinking of getting", "should i get", "checkout",
    "add to cart", "grab", "treat myself", "splurge", "just bought",
    "just ordered", "just spent", "gonna buy", "thinking of buying",
    "want to order", "about to buy", "going to buy",
]

HIGH_STRESS_EMOTIONS = [
    "frustrated", "angry", "sad", "fearful",
    "tired", "anxious", "stressed", "lonely", "confused",
]

SPEND_CATEGORIES = [
    "Food & Dining", "Shopping", "Entertainment",
    "Travel", "Health", "Technology", "Subscriptions", "Other",
]

# ══════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def safety_check(text: str) -> str:
    crisis = [
        "hurt myself", "end it", "give up", "can't go on",
        "want to die", "kill myself", "no point", "disappear",
    ]
    urgent = ["emergency", "call 911", "ambulance", "i'm in danger"]
    t = text.lower()
    if any(s in t for s in crisis):
        logger.warning("Safety: crisis signal detected")
        return (
            "I hear you — you're going through something really hard. "
            "Please reach out to someone you trust or a crisis line. You're not alone."
        )
    if any(s in t for s in urgent):
        logger.warning("Safety: urgent signal detected")
        return (
            "This sounds urgent — please contact emergency services immediately. "
            "Get real help first."
        )
    return ""


def check_intervention(text: str, emotion: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SPEND_KEYWORDS) and emotion in HIGH_STRESS_EMOTIONS


def get_emotional_spend_history(emotion: str) -> dict:
    """Returns longitudinal spend data for a specific emotional state."""
    if not db:
        return {"count": 0, "total": 0.0, "avg": 0.0, "examples": [], "avg_regret": None}
    try:
        docs = (
            db.collection("spend_logs")
            .where("emotion_hint", "==", emotion)
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(10)
            .get()
        )
        data = [d.to_dict() for d in docs]
        total = sum(d.get("amount", 0) for d in data)
        count = len(data)
        examples = [f"${d['amount']:.0f} on {d.get('category', '?')}" for d in data[:3]]

        # Compute average regret for this emotion
        regret_scores = [d["regret_score"] for d in data if d.get("regret_score") is not None]
        avg_regret = round(sum(regret_scores) / len(regret_scores), 1) if regret_scores else None

        return {
            "count": count,
            "total": round(total, 2),
            "avg": round(total / count, 2) if count else 0,
            "examples": examples,
            "avg_regret": avg_regret,
        }
    except Exception as e:
        logger.error(f"Emotional history failed: {e}")
        return {"count": 0, "total": 0.0, "avg": 0.0, "examples": [], "avg_regret": None}


def compute_vulnerability_score(emotion: str) -> float:
    """
    Returns a 0–10 vulnerability score combining spend frequency,
    average amount, and average regret for the given emotion.
    Higher = more dangerous to spend in this state.
    """
    hist = get_emotional_spend_history(emotion)
    if hist["count"] == 0:
        return 0.0
    freq_factor = min(hist["count"] / 10, 1.0) * 4       # up to 4 points for frequency
    amt_factor  = min(hist["avg"] / 250, 1.0) * 3        # up to 3 points for avg amount
    regret_factor = (hist["avg_regret"] / 10 * 3) if hist["avg_regret"] else 1.5  # up to 3 pts
    score = freq_factor + amt_factor + regret_factor
    return round(min(score, 10.0), 1)


def analyze_emotion_from_frame(image_base64: str) -> str:
    try:
        return _vision(
            image_base64,
            (
                "Describe this person's emotional state in ONE sentence. "
                "Format: 'User appears [emotion], [one specific observation].' "
                "If no face visible: 'Visual context unclear.'"
            ),
        )
    except Exception as e:
        logger.error(f"Vision failed: {e}")
        return "Visual context unavailable."


def generate_session_title(first_message: str) -> str:
    try:
        return _chat(
            [{"role": "user", "content": (
                f"Create a 3-5 word title for a financial conversation that starts with: "
                f"'{first_message[:80]}'. Return ONLY the title, no punctuation."
            )}],
            max_tokens=20,
        )
    except Exception:
        return first_message[:30] + ("..." if len(first_message) > 30 else "")


def save_turn(
    sid: str,
    session: dict,
    user_text: str,
    ai_response: str,
    emotion: str,
    visual_context: str,
    intervention: bool = False,
    vulnerability_score: float = 0.0,
):
    if not db:
        return
    try:
        db.collection("sessions").document(sid).collection("turns").add({
            "user_message": user_text,
            "ai_response": ai_response,
            "emotion_hint": emotion,
            "visual_context": visual_context,
            "intervention_triggered": intervention,
            "vulnerability_score": vulnerability_score,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "turn_number": len(session["conversation_history"]) // 2,
        })
        db.collection("sessions").document(sid).set(
            {
                "title": session["session_title"] or user_text[:40],
                "total_turns": len(session["conversation_history"]) // 2,
                "last_active": firestore.SERVER_TIMESTAMP,
                "preview": user_text[:80],
                "type": "finance",
            },
            merge=True,
        )
    except Exception as e:
        logger.error(f"Firestore save failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "status": "Sentience Finance running",
        "version": "2.0",
        "active_sessions": len(_sessions),
        "memory": "Firestore" if db else "in-memory",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "firebase": db is not None,
        "groq": bool(os.getenv("GROQ_API_KEY")),
        "version": "2.0",
        "active_sessions": len(_sessions),
    }


# ── SPEAK (main multimodal inference) ──────────────────────────────
@app.post("/speak")
@limiter.limit("30/minute")
def speak(
    request: Request,
    message: Message,
    x_session_id: Optional[str] = Header(default=None),
):
    # ── Resolve session ──
    sid = x_session_id or str(uuid.uuid4())
    session = get_session(sid)
    conv = session["conversation_history"]

    logger.info(
        f"[{sid[:8]}] Turn {len(conv)//2+1} | "
        f"emotion={message.emotion_hint} | vision={'yes' if message.image_frame else 'no'}"
    )

    # ── Safety gate ──
    safe = safety_check(message.text)
    if safe:
        return {
            "response": safe,
            "status": "safety_intercept",
            "visual_context": "",
            "session_id": sid,
            "turns": len(conv) // 2,
        }

    # ── Auto-title (first turn only) ──
    if not session["session_title"] and not conv:
        session["session_title"] = generate_session_title(message.text)

    # ── Vision ──
    visual_context = ""
    if message.image_frame:
        visual_context = analyze_emotion_from_frame(message.image_frame)

    # ── Effective emotion (camera overrides chip) ──
    effective_emotion = message.emotion_hint
    if visual_context and visual_context not in (
        "Visual context unavailable.", "Visual context unclear."
    ):
        vc_lower = visual_context.lower()
        for emo in HIGH_STRESS_EMOTIONS + ["happy", "calm", "neutral", "surprised"]:
            if emo in vc_lower:
                effective_emotion = emo
                break

    # ── Vulnerability score for this moment ──
    vuln_score = compute_vulnerability_score(effective_emotion)

    # ── Intervention logic ──
    should_intervene = check_intervention(message.text, effective_emotion)
    intervention_block = ""
    if should_intervene:
        hist = get_emotional_spend_history(effective_emotion)
        logger.info(
            f"[{sid[:8]}] INTERVENTION | emotion={effective_emotion} | "
            f"past_count={hist['count']} | vuln={vuln_score}"
        )
        if hist["count"] > 0:
            ex = ", ".join(hist["examples"]) if hist["examples"] else "similar purchases"
            regret_line = (
                f"Average regret score after these purchases: {hist['avg_regret']}/10."
                if hist["avg_regret"] is not None else ""
            )
            intervention_block = (
                f"\n[INTERVENTION] CRITICAL MOMENT. "
                f"User is {effective_emotion} and signaling purchase intent. "
                f"Their OWN data shows: {hist['count']} purchases while {effective_emotion}, "
                f"${hist['total']:.0f} total spent, avg ${hist['avg']:.0f} per transaction. "
                f"Recent examples: {ex}. {regret_line} "
                f"Vulnerability score right now: {vuln_score}/10. "
                f"Start your response with 'Hold on.' "
                f"Surface their EXACT numbers compassionately. "
                f"Suggest naming this feeling. Suggest 24-hour pause. "
                f"End with exactly ONE question. Never shame them."
            )
        else:
            intervention_block = (
                f"\n[INTERVENTION] User is {effective_emotion} and considering a purchase. "
                f"No purchase history yet in this emotional state — this could be the start of a pattern. "
                f"Acknowledge their emotion warmly. Ask: is now the right time?"
            )

    # ── Build user message content ──
    vc_valid = visual_context and visual_context not in (
        "Visual context unavailable.", "Visual context unclear."
    )
    if vc_valid:
        uc = f"[Visual: {visual_context}] [Tone: {effective_emotion}]{intervention_block}\n{message.text}"
    elif effective_emotion != "neutral":
        uc = f"[Tone: {effective_emotion}]{intervention_block}\n{message.text}"
    elif intervention_block:
        uc = f"{intervention_block}\n{message.text}"
    else:
        uc = message.text

    conv.append({"role": "user", "content": uc})
    if len(conv) > 20:
        session["conversation_history"] = conv[-20:]
        conv = session["conversation_history"]

    # ── LLM call ──
    try:
        ai_resp = _chat(
            [{"role": "system", "content": SENTIENCE_SYSTEM_PROMPT}] + conv,
            max_tokens=300,
        )
        conv.append({"role": "assistant", "content": ai_resp})
        save_turn(
            sid, session, message.text, ai_resp,
            effective_emotion, visual_context, should_intervene, vuln_score,
        )

        return {
            "response": ai_resp,
            "visual_context": visual_context,
            "status": "intervention" if should_intervene else "success",
            "intervention": should_intervene,
            "emotion_detected": effective_emotion,
            "vulnerability_score": vuln_score,
            "session_id": sid,
            "session_title": session["session_title"],
            "turns": len(conv) // 2,
        }
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return {
            "response": "Having trouble connecting — please try again in a moment.",
            "status": "error",
            "error": str(e),
            "session_id": sid,
        }


# ── LOG SPEND ──────────────────────────────────────────────────────
@app.post("/log-spend")
@limiter.limit("60/minute")
def log_spend(
    request: Request,
    spend: SpendLog,
    x_session_id: Optional[str] = Header(default=None),
):
    if not db:
        return {"status": "error", "error": "Firebase not configured"}
    sid = x_session_id or "unknown"
    try:
        spend_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        db.collection("spend_logs").document(spend_id).set({
            "amount": spend.amount,
            "category": spend.category,
            "description": spend.description,
            "emotion_hint": spend.emotion_hint,
            "visual_context": spend.visual_context,
            "session_id": sid,
            "regret_score": None,
            "timestamp": firestore.SERVER_TIMESTAMP,
            "spend_id": spend_id,
        })
        logger.info(f"Spend: ${spend.amount} | {spend.category} | {spend.emotion_hint}")

        try:
            ai_comment = _chat(
                [
                    {"role": "system", "content": SENTIENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"User logged: ${spend.amount} on {spend.category} "
                        f"({spend.description or 'no description'}) while feeling {spend.emotion_hint}. "
                        f"Give one warm 1-2 sentence acknowledgment that notes the emotional context "
                        f"and hints at watching this pattern."
                    )},
                ],
                max_tokens=80,
            )
        except Exception:
            ai_comment = (
                f"Logged ${spend.amount} on {spend.category} while {spend.emotion_hint}. "
                f"I'll watch this pattern with you."
            )

        return {"status": "logged", "spend_id": spend_id, "ai_comment": ai_comment}
    except Exception as e:
        logger.error(f"Log spend failed: {e}")
        return {"status": "error", "error": str(e)}


# ── REGRET SCORE ───────────────────────────────────────────────────
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


# ── INSIGHTS (Psychological Ledger) ───────────────────────────────
@app.get("/insights")
def get_insights():
    if not db:
        return {
            "chart_data": [], "category_data": [], "recent_spends": [],
            "total_logged": 0, "narrative": "Firebase not configured.",
            "danger_emotion": None, "vulnerability_map": {},
        }
    try:
        docs = (
            db.collection("spend_logs")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(200)
            .get()
        )

        emotion_data: dict = {}
        category_data: dict = {}
        total_logged = 0.0
        recent = []

        for doc in docs:
            d = doc.to_dict()
            emotion   = d.get("emotion_hint", "neutral")
            amount    = d.get("amount", 0)
            category  = d.get("category", "Other")
            regret    = d.get("regret_score")
            ts        = d.get("timestamp")
            total_logged += amount

            if emotion not in emotion_data:
                emotion_data[emotion] = {
                    "total": 0, "count": 0,
                    "regret_sum": 0, "regret_count": 0,
                }
            emotion_data[emotion]["total"]  += amount
            emotion_data[emotion]["count"]  += 1
            if regret is not None:
                emotion_data[emotion]["regret_sum"]   += regret
                emotion_data[emotion]["regret_count"] += 1

            category_data[category] = category_data.get(category, 0) + amount

            if len(recent) < 10:
                recent.append({
                    "amount": amount, "category": category,
                    "emotion": emotion,
                    "description": d.get("description", ""),
                    "regret_score": regret,
                    "spend_id": d.get("spend_id", doc.id),
                    "timestamp": ts.isoformat() if hasattr(ts, "isoformat") else None,
                })

        # Build chart_data with vulnerability scores
        chart_data = []
        vulnerability_map: dict[str, float] = {}
        for emo, data in emotion_data.items():
            avg_regret = (
                round(data["regret_sum"] / data["regret_count"], 1)
                if data["regret_count"] > 0 else None
            )
            vuln = compute_vulnerability_score(emo)
            vulnerability_map[emo] = vuln
            chart_data.append({
                "emotion":         emo,
                "total":           round(data["total"], 2),
                "count":           data["count"],
                "avg_spend":       round(data["total"] / data["count"], 2),
                "avg_regret":      avg_regret,
                "vulnerability":   vuln,
            })
        chart_data.sort(key=lambda x: x["total"], reverse=True)

        # ── Danger Emotion: highest avg_regret with ≥2 data points ──
        candidates = [
            (emo, d["regret_sum"] / d["regret_count"])
            for emo, d in emotion_data.items()
            if d["regret_count"] >= 2
        ]
        danger_emotion = max(candidates, key=lambda x: x[1])[0] if candidates else (
            chart_data[0]["emotion"] if chart_data else None
        )
        danger_stats = emotion_data.get(danger_emotion, {}) if danger_emotion else {}
        danger_avg_regret = (
            round(danger_stats["regret_sum"] / danger_stats["regret_count"], 1)
            if danger_stats.get("regret_count", 0) > 0 else None
        )

        # ── AI Narrative ──
        narrative = "Start logging expenses to see your emotional spending patterns."
        if chart_data:
            try:
                emotion_summary  = str([{x["emotion"]: f"${x['total']:.0f}"} for x in chart_data[:4]])
                category_summary = str(
                    dict(
                        list(sorted(category_data.items(), key=lambda x: x[1], reverse=True))[:4]
                    )
                )
                narrative = _chat(
                    [{"role": "user", "content": (
                        f"Give ONE honest 2-sentence behavioral finance insight about this data. "
                        f"Total logged: ${total_logged:.0f}. "
                        f"Spending by emotion: {emotion_summary}. "
                        f"Spending by category: {category_summary}. "
                        f"Danger emotion (highest regret): {danger_emotion}. "
                        f"Be specific, human, and memorable. No fluff. No lists."
                    )}],
                    max_tokens=90,
                )
            except Exception as e:
                logger.error(f"Narrative generation failed: {e}")
                top = chart_data[0]
                narrative = (
                    f"You spend most when feeling {top['emotion']} "
                    f"— ${top['total']:.0f} across {top['count']} transactions."
                )

        return {
            "chart_data": chart_data,
            "category_data": [
                {"category": k, "total": round(v, 2)}
                for k, v in sorted(category_data.items(), key=lambda x: x[1], reverse=True)
            ],
            "recent_spends":      recent,
            "total_logged":       round(total_logged, 2),
            "narrative":          narrative,
            "danger_emotion":     danger_emotion,
            "danger_avg_regret":  danger_avg_regret,
            "vulnerability_map":  vulnerability_map,
        }
    except Exception as e:
        logger.error(f"Insights failed: {e}")
        return {
            "chart_data": [], "category_data": [], "recent_spends": [],
            "total_logged": 0, "narrative": "Could not load insights.",
            "danger_emotion": None, "vulnerability_map": {},
            "error": str(e),
        }


# ── PATTERN REPORT (new — for demo "Danger Emotion" card) ──────────
@app.get("/pattern-report/{emotion}")
def pattern_report(emotion: str):
    """
    Returns a vulnerability profile for a specific emotional state.
    Used by the frontend Vulnerability Meter widget.
    """
    hist = get_emotional_spend_history(emotion)
    vuln = compute_vulnerability_score(emotion)

    if hist["count"] == 0:
        return {
            "emotion": emotion,
            "vulnerability_score": 0,
            "verdict": "No data yet",
            "total_at_risk": 0,
            "avg_per_transaction": 0,
            "avg_regret": None,
            "transaction_count": 0,
            "examples": [],
        }

    # Human-readable verdict
    if vuln >= 7:
        verdict = "High Risk — historically you overspend and regret it in this state"
    elif vuln >= 4:
        verdict = "Moderate Risk — proceed with caution"
    else:
        verdict = "Low Risk — you tend to make reasonable decisions here"

    return {
        "emotion":              emotion,
        "vulnerability_score":  vuln,
        "verdict":              verdict,
        "total_at_risk":        hist["total"],
        "avg_per_transaction":  hist["avg"],
        "avg_regret":           hist["avg_regret"],
        "transaction_count":    hist["count"],
        "examples":             hist["examples"],
    }


# ── PROACTIVE CHECK-IN ─────────────────────────────────────────────
@app.get("/check-in")
@limiter.limit("10/minute")
def check_in(
    request: Request,
    x_session_id: Optional[str] = Header(default=None),
):
    sid = x_session_id or ""
    session = _sessions.get(sid, {})
    conv = session.get("conversation_history", [])
    snippet = conv[-3:] if conv else []
    try:
        msg = _chat(
            [
                {"role": "system", "content": SENTIENCE_SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"User has been quiet for 30 seconds. "
                    f"Recent context: {snippet if snippet else 'new session'}. "
                    f"Send ONE proactive financial check-in under 2 sentences. "
                    f"Options: ask how they feel about money today, "
                    f"remind them to log a recent purchase, or share a quick behavioral insight."
                )},
            ],
            max_tokens=80,
        )
        return {"message": msg, "type": "proactive"}
    except Exception as e:
        logger.error(f"Check-in failed: {e}")
        return {
            "message": "How are you feeling about your finances today?",
            "type": "proactive",
        }


# ── RESET ──────────────────────────────────────────────────────────
@app.post("/reset")
def reset(x_session_id: Optional[str] = Header(default=None)):
    new_sid = str(uuid.uuid4())
    if x_session_id and x_session_id in _sessions:
        del _sessions[x_session_id]
    get_session(new_sid)  # pre-create the new session
    return {"status": "reset", "new_session_id": new_sid}


# ── SESSION HISTORY ────────────────────────────────────────────────
@app.get("/sessions")
def get_sessions():
    if not db:
        return {"sessions": [], "error": "Firebase not configured"}
    try:
        docs = (
            db.collection("sessions")
            .order_by("last_active", direction=firestore.Query.DESCENDING)
            .limit(30)
            .get()
        )
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
        docs = (
            db.collection("sessions")
            .document(sid)
            .collection("turns")
            .order_by("turn_number")
            .get()
        )
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
def load_session(sid: str, x_session_id: Optional[str] = Header(default=None)):
    if not db:
        return {"status": "error", "error": "Firebase not configured"}
    try:
        s = db.collection("sessions").document(sid).get()
        if not s.exists:
            return {"status": "error", "error": "Session not found"}
        sd = s.to_dict()
        docs = (
            db.collection("sessions")
            .document(sid)
            .collection("turns")
            .order_by("turn_number")
            .get()
        )
        restored = []
        for t in docs:
            d = t.to_dict()
            um  = d.get("user_message", "")
            vis = d.get("visual_context", "")
            emo = d.get("emotion_hint", "neutral")
            if vis and vis not in ("Visual context unavailable.", "Visual context unclear."):
                uc = f"[Visual: {vis}] [Tone: {emo}]\n{um}"
            elif emo != "neutral":
                uc = f"[Tone: {emo}]\n{um}"
            else:
                uc = um
            restored.append({"role": "user",      "content": uc})
            restored.append({"role": "assistant", "content": d.get("ai_response", "")})

        # Load into the caller's session slot (create new if needed)
        target_sid = x_session_id or sid
        session = get_session(target_sid)
        session["conversation_history"] = restored[-20:]
        session["session_title"] = sd.get("title", sid)

        return {
            "status": "loaded",
            "session_id": target_sid,
            "session_title": session["session_title"],
            "turns_loaded": len(restored) // 2,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════════
#  GEMINI LIVE WebSocket  —  /ws/live
#  ─────────────────────────────────────────────────────────────────
#  Architecture:
#    Browser  ──PCM 16kHz──▶  FastAPI WS  ──▶  Gemini Live API
#    Browser  ◀──PCM 24kHz──  FastAPI WS  ◀──  Gemini Live API
#
#  Context Injection:
#    On connect, we query Firestore for this user's vulnerability
#    profile and inject it as a system prompt so the voice version
#    of Sentience has the SAME memory as the text version.
#
#  Barge-in / Interrupt:
#    Frontend sends {"type": "interrupt"} to stop playback.
#    We set a flag that drains the outgoing audio queue silently.
#
#  Intervention Detection:
#    Every transcript chunk is checked for spend keywords + emotion.
#    If triggered, we inject an [INTERVENTION] signal into the
#    system context for the current turn.
# ══════════════════════════════════════════════════════════════════

GEMINI_LIVE_MODEL = "gemini-2.0-flash-live-001"

def _build_live_system_prompt(emotion: str, vuln_score: float,
                               hist: dict, danger_emotion: Optional[str]) -> str:
    """
    Builds a context-rich system prompt for Gemini Live that includes
    the user's real behavioral data from Firestore.
    """
    behavior_block = ""
    if hist["count"] > 0:
        ex = ", ".join(hist["examples"]) if hist["examples"] else "similar items"
        regret_line = (
            f"When they later rated those purchases, average regret was {hist['avg_regret']}/10."
            if hist["avg_regret"] is not None else ""
        )
        behavior_block = (
            f"\n\nUSER BEHAVIORAL PROFILE (from their own data):\n"
            f"Current emotion: {emotion}\n"
            f"Vulnerability score right now: {vuln_score}/10\n"
            f"Past purchases while {emotion}: {hist['count']} transactions, "
            f"${hist['total']:.0f} total, avg ${hist['avg']:.0f} each.\n"
            f"Recent examples: {ex}. {regret_line}\n"
            f"Danger emotion (highest regret state): {danger_emotion or 'not yet determined'}.\n"
            f"\nIf you hear them mention buying, ordering, or spending: "
            f"reference THEIR OWN numbers. Say 'Hold on.' first. "
            f"Be warm, never shame them."
        )
    else:
        behavior_block = (
            f"\n\nUSER BEHAVIORAL PROFILE:\n"
            f"Current emotion: {emotion} | Vulnerability: {vuln_score}/10\n"
            f"No spending history yet — this is their first session. "
            f"Help them start building self-awareness."
        )

    return SENTIENCE_SYSTEM_PROMPT.strip() + behavior_block


def _get_danger_emotion_from_db() -> Optional[str]:
    """Fast Firestore query to find the emotion with highest average regret."""
    if not db:
        return None
    try:
        docs = (db.collection("spend_logs")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(100).get())
        emotion_regrets: dict = {}
        for doc in docs:
            d = doc.to_dict()
            emo = d.get("emotion_hint")
            regret = d.get("regret_score")
            if emo and regret is not None:
                if emo not in emotion_regrets:
                    emotion_regrets[emo] = []
                emotion_regrets[emo].append(regret)
        best, best_avg = None, -1.0
        for emo, scores in emotion_regrets.items():
            if len(scores) >= 2:
                avg = sum(scores) / len(scores)
                if avg > best_avg:
                    best_avg = avg
                    best = emo
        return best
    except Exception:
        return None


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Real-time bidirectional audio bridge between browser and Gemini Live API.

    Message protocol (JSON from browser → backend):
      {"type": "init", "session_id": "uuid", "emotion": "stressed"}
      {"type": "audio", "data": "<base64 PCM 16kHz mono>"}
      {"type": "emotion", "data": "frustrated"}   ← chip/camera update mid-session
      {"type": "interrupt"}                         ← barge-in: stop current playback
      {"type": "end_turn"}                          ← user stopped speaking

    Message protocol (JSON from backend → browser):
      {"type": "status", "data": "connected"}
      {"type": "text", "data": "transcript text chunk"}
      {"type": "audio", "data": "<base64 PCM 24kHz>", "mime_type": "audio/pcm;rate=24000"}
      {"type": "turn_complete"}
      {"type": "intervention", "data": {"score": 7.2, "emotion": "stressed"}}
      {"type": "error", "data": "error message"}
    """
    await websocket.accept()
    logger.info("WS /live: connection accepted")

    # ── Check for Google API key ──────────────────────────────────
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        await websocket.send_json({
            "type": "error",
            "data": "GOOGLE_API_KEY not configured on server. Set it in Render env vars."
        })
        await websocket.close()
        return

    # ── Wait for init message ─────────────────────────────────────
    try:
        raw_init = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
    except asyncio.TimeoutError:
        await websocket.send_json({"type": "error", "data": "Init timeout"})
        await websocket.close()
        return

    session_id = raw_init.get("session_id", str(uuid.uuid4()))
    current_emotion = raw_init.get("emotion", "neutral")
    logger.info(f"WS /live: session={session_id[:8]} emotion={current_emotion}")

    # ── Load behavioral context from Firestore ────────────────────
    hist = get_emotional_spend_history(current_emotion)
    vuln_score = compute_vulnerability_score(current_emotion)
    danger_emotion = _get_danger_emotion_from_db()
    system_prompt = _build_live_system_prompt(
        current_emotion, vuln_score, hist, danger_emotion
    )
    logger.info(
        f"WS /live: context loaded | vuln={vuln_score} | "
        f"danger_emo={danger_emotion} | hist_count={hist['count']}"
    )

    # ── Shared state ──────────────────────────────────────────────
    interrupting = False          # True while we're draining old audio
    full_transcript = ""          # accumulates turn transcript for intervention check

    # ── Connect to Gemini Live ────────────────────────────────────
    try:
        from google import genai as google_genai
        from google.genai import types as genai_types

        g_client = google_genai.Client(api_key=google_api_key)

        live_config = genai_types.LiveConnectConfig(
            response_modalities=["AUDIO", "TEXT"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name="Aoede"   # warm female voice — matches Sentience brand
                    )
                )
            ),
            system_instruction=genai_types.Content(
                parts=[genai_types.Part(text=system_prompt)]
            ),
        )

        async with g_client.aio.live.connect(
            model=GEMINI_LIVE_MODEL,
            config=live_config,
        ) as live_session:

            await websocket.send_json({"type": "status", "data": "connected"})
            logger.info("WS /live: Gemini Live session established")

            # ── Task 1: Browser → Gemini (audio sender) ────────────
            async def send_loop():
                nonlocal current_emotion, interrupting, full_transcript, \
                         hist, vuln_score, system_prompt

                while True:
                    try:
                        msg = await websocket.receive_json()
                    except WebSocketDisconnect:
                        logger.info("WS /live: browser disconnected (send_loop)")
                        break
                    except Exception:
                        break

                    msg_type = msg.get("type")

                    # ── Raw audio chunk ──────────────────────────────
                    if msg_type == "audio":
                        raw_b64 = msg.get("data", "")
                        if not raw_b64:
                            continue
                        try:
                            pcm_bytes = base64.b64decode(raw_b64)
                            await live_session.send(
                                input=genai_types.LiveClientRealtimeInput(
                                    media_chunks=[
                                        genai_types.Blob(
                                            mime_type="audio/pcm;rate=16000",
                                            data=pcm_bytes,
                                        )
                                    ]
                                )
                            )
                        except Exception as e:
                            logger.error(f"WS /live: audio send error: {e}")

                    # ── Emotion chip update mid-session ──────────────
                    elif msg_type == "emotion":
                        new_emotion = msg.get("data", "neutral")
                        if new_emotion != current_emotion:
                            current_emotion = new_emotion
                            hist = get_emotional_spend_history(current_emotion)
                            vuln_score = compute_vulnerability_score(current_emotion)
                            system_prompt = _build_live_system_prompt(
                                current_emotion, vuln_score, hist, danger_emotion
                            )
                            logger.info(f"WS /live: emotion updated → {current_emotion} vuln={vuln_score}")
                            # Notify browser of updated vulnerability
                            await websocket.send_json({
                                "type": "vulnerability_update",
                                "data": {
                                    "emotion": current_emotion,
                                    "score": vuln_score,
                                    "verdict": (
                                        "High risk — pause before buying" if vuln_score >= 7
                                        else "Moderate — proceed carefully" if vuln_score >= 4
                                        else "Low risk in this state"
                                    )
                                }
                            })

                    # ── Barge-in: user interrupted AI ────────────────
                    elif msg_type == "interrupt":
                        interrupting = True
                        full_transcript = ""   # reset turn transcript
                        logger.info("WS /live: barge-in interrupt received")

                    # ── End of user turn ─────────────────────────────
                    elif msg_type == "end_turn":
                        full_transcript = ""   # reset for next turn

            # ── Task 2: Gemini → Browser (receiver) ───────────────
            async def receive_loop():
                nonlocal interrupting, full_transcript

                async for response in live_session.receive():
                    # ── Barge-in drain ───────────────────────────────
                    if interrupting:
                        # Check if Gemini itself acknowledged the interrupt
                        # (server_content.interrupted = True signals this)
                        if (hasattr(response, "server_content") and
                                response.server_content and
                                getattr(response.server_content, "interrupted", False)):
                            interrupting = False
                            logger.info("WS /live: interrupt acknowledged by Gemini")
                        # While draining, skip audio but pass text for context
                        if response.data:
                            continue   # drop stale audio frames

                    # ── Audio response ───────────────────────────────
                    if response.data:
                        encoded = base64.b64encode(response.data).decode("utf-8")
                        try:
                            await websocket.send_json({
                                "type": "audio",
                                "data": encoded,
                                "mime_type": "audio/pcm;rate=24000",
                            })
                        except Exception:
                            break  # client disconnected

                    # ── Text / transcript chunk ──────────────────────
                    if response.text:
                        full_transcript += response.text
                        try:
                            await websocket.send_json({
                                "type": "text",
                                "data": response.text,
                            })
                        except Exception:
                            break

                        # ── Intervention detection on transcript ─────
                        if check_intervention(full_transcript, current_emotion):
                            try:
                                await websocket.send_json({
                                    "type": "intervention",
                                    "data": {
                                        "score": vuln_score,
                                        "emotion": current_emotion,
                                        "history": hist["examples"][:3],
                                    }
                                })
                            except Exception:
                                pass

                    # ── Turn complete ────────────────────────────────
                    if (hasattr(response, "server_content") and
                            response.server_content and
                            getattr(response.server_content, "turn_complete", False)):
                        interrupting = False  # safe to receive new audio
                        try:
                            await websocket.send_json({"type": "turn_complete"})
                        except Exception:
                            break
                        # Save turn to Firestore if we have a meaningful transcript
                        if full_transcript.strip():
                            session_obj = get_session(session_id)
                            save_turn(
                                sid=session_id,
                                session=session_obj,
                                user_text="[voice]",
                                ai_response=full_transcript.strip(),
                                emotion=current_emotion,
                                visual_context="",
                                intervention=check_intervention(full_transcript, current_emotion),
                                vulnerability_score=vuln_score,
                            )
                        full_transcript = ""

            # ── Run both loops concurrently ───────────────────────
            send_task    = asyncio.create_task(send_loop())
            receive_task = asyncio.create_task(receive_loop())
            done, pending = await asyncio.wait(
                [send_task, receive_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            logger.info(f"WS /live: session {session_id[:8]} ended")

    except ImportError:
        logger.error("google-genai SDK not installed")
        try:
            await websocket.send_json({
                "type": "error",
                "data": "google-genai SDK missing. Run: pip install google-genai>=1.0.0"
            })
        except Exception:
            pass

    except Exception as e:
        logger.error(f"WS /live: fatal error: {e}")
        try:
            await websocket.send_json({"type": "error", "data": str(e)})
        except Exception:
            pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info(f"WS /live: closed — session {session_id[:8]}")