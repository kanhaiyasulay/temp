# main.py
# -------------------------------------------------------------------------
# FastAPI backend for Voice Mode Agent — Avatar Edition
# -------------------------------------------------------------------------
# PyAudio is removed. All real-time audio/video happens in the browser
# via Azure Speech SDK + WebRTC.  This backend provides:
#   GET  /api/speech-token  — short-lived token + ICE creds for the avatar
#   POST /api/chat          — relay user text to Azure OpenAI, return reply
#   POST /api/evaluate      — evaluate a conversation transcript
# -------------------------------------------------------------------------

import json
import os

import truststore
truststore.inject_into_ssl()

import httpx
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=True)

app = FastAPI(title="Voice Mode Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
CONVERSATION_FILE = os.path.join(BACKEND_DIR, "logs", "last_conversation.json")


# ── Request models ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    user_message: str
    conversation_history: list[dict] = []
    system_prompt: str = ""


class EvaluateRequest(BaseModel):
    transcript: list[dict]


# ── GET /api/speech-token ───────────────────────────────────────────────

@app.get("/api/speech-token")
async def get_speech_token():
    """Exchange the Speech API key for a short-lived auth token and
    ICE relay credentials needed by the browser for WebRTC avatar."""
    speech_key = os.getenv("AZURE_SPEECH_KEY", "")
    speech_region = os.getenv("AZURE_SPEECH_REGION", "")

    if not speech_key or not speech_region:
        raise HTTPException(
            status_code=500,
            detail="AZURE_SPEECH_KEY and AZURE_SPEECH_REGION must be set in .env",
        )

    # 1) Speech authorization token (valid ~10 min)
    token_url = (
        f"https://{speech_region}.api.cognitive.microsoft.com"
        "/sts/v1.0/issueToken"
    )
    try:
        r = requests.post(
            token_url,
            headers={"Ocp-Apim-Subscription-Key": speech_key},
            timeout=10,
        )
        r.raise_for_status()
        speech_token = r.text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Token request failed: {exc}")

    # 2) ICE relay credentials for avatar WebRTC
    ice_url = (
        f"https://{speech_region}.tts.speech.microsoft.com"
        "/cognitiveservices/avatar/relay/token/v1"
    )
    try:
        r = requests.get(
            ice_url,
            headers={"Ocp-Apim-Subscription-Key": speech_key},
            timeout=10,
        )
        r.raise_for_status()
        ice_data = r.json()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"ICE token request failed: {exc}")

    return {
        "token": speech_token,
        "region": speech_region,
        "iceServers": ice_data,
        "avatarCharacter": os.getenv("AZURE_AVATAR_CHARACTER", "lisa"),
        "avatarStyle": os.getenv("AZURE_AVATAR_STYLE", "casual-sitting"),
        "voiceName": os.getenv(
            "AZURE_SPEECH_VOICE", "en-US-AvaMultilingualNeural"
        ),
    }


# ── POST /api/chat ─────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send user text to Azure OpenAI and return the AI customer reply."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_KEY", "")
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    if not endpoint or not api_key:
        raise HTTPException(
            status_code=500, detail="Azure OpenAI credentials not configured"
        )

    system_prompt = (
        "You are pretending to be a potential customer evaluating a product.\n\n"
        "Your personality:\n"
        "- Curious\n"
        "- Slightly skeptical\n"
        "- Ask realistic questions\n"
        "- Raise objections about price, ROI, competitors\n\n"
        "Speak naturally like a human customer. "
        "Keep responses conversational and concise (2-4 sentences)."
    )
    if req.system_prompt:
        system_prompt += f"\n\nAdditional context:\n{req.system_prompt}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.conversation_history:
        messages.append(
            {
                "role": "assistant" if msg.get("role") == "assistant" else "user",
                "content": msg["text"],
            }
        )
    messages.append({"role": "user", "content": req.user_message})

    url = (
        f"{endpoint}/openai/deployments/{model}"
        "/chat/completions?api-version=2024-12-01-preview"
    )
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body = {"messages": messages, "temperature": 0.8, "max_tokens": 150}

    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        return {"response": r.json()["choices"][0]["message"]["content"]}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"OpenAI error: {exc}")


# ── GET /api/chat-stream (SSE) ─────────────────────────────────────────

@app.post("/api/chat-stream")
async def chat_stream(req: ChatRequest):
    """Stream Azure OpenAI tokens as SSE so the frontend can start avatar
    speech on the first sentence without waiting for the full response."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_KEY", "")
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    if not endpoint or not api_key:
        raise HTTPException(
            status_code=500, detail="Azure OpenAI credentials not configured"
        )

    system_prompt = (
        "You are pretending to be a potential customer evaluating a product.\n\n"
        "Your personality:\n"
        "- Curious\n"
        "- Slightly skeptical\n"
        "- Ask realistic questions\n"
        "- Raise objections about price, ROI, competitors\n\n"
        "Speak naturally like a human customer. "
        "Keep responses conversational and concise (1-2 sentences)."
    )
    if req.system_prompt:
        system_prompt += f"\n\nAdditional context:\n{req.system_prompt}"

    messages = [{"role": "system", "content": system_prompt}]
    for msg in req.conversation_history:
        messages.append(
            {
                "role": "assistant" if msg.get("role") == "assistant" else "user",
                "content": msg["text"],
            }
        )
    messages.append({"role": "user", "content": req.user_message})

    url = (
        f"{endpoint}/openai/deployments/{model}"
        "/chat/completions?api-version=2024-12-01-preview"
    )
    headers_oai = {"api-key": api_key, "Content-Type": "application/json"}
    body = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 150,
        "stream": True,
    }

    async def _generate():
        async with httpx.AsyncClient(timeout=30) as client:
            async with client.stream(
                "POST", url, headers=headers_oai, json=body
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload.strip() == "[DONE]":
                        yield "data: [DONE]\n\n"
                        return
                    try:
                        chunk = json.loads(payload)
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── POST /api/evaluate ─────────────────────────────────────────────────

@app.post("/api/evaluate")
async def evaluate(req: EvaluateRequest):
    """Evaluate a sales conversation transcript and return structured feedback."""
    _save_conversation(req.transcript)
    return _generate_evaluation(req.transcript)


# ── Helpers ─────────────────────────────────────────────────────────────

def _save_conversation(transcript: list[dict]):
    os.makedirs(os.path.join(BACKEND_DIR, "logs"), exist_ok=True)
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


def _generate_evaluation(conversation: list[dict]) -> dict:
    """Call Azure OpenAI to evaluate the sales conversation."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_KEY", "")
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini")

    if not conversation:
        return {
            "strength": ["No conversation recorded."],
            "weakness": [],
            "rating": 0,
            "summary": "The session ended before any conversation took place.",
        }

    transcript_text = "\n".join(
        f"{m['role']}: {m['text']}" for m in conversation
    )

    url = (
        f"{endpoint}/openai/deployments/{model}"
        "/chat/completions?api-version=2024-12-01-preview"
    )
    headers = {"api-key": api_key, "Content-Type": "application/json"}
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert sales trainer. Evaluate the salesperson "
                    "based on the conversation transcript below.\n\n"
                    "Respond ONLY in valid JSON format:\n"
                    "{\n"
                    '  "strength": ["point1", "point2", "point3"],\n'
                    '  "weakness": ["point1", "point2", "point3"],\n'
                    '  "rating": 7.5,\n'
                    '  "summary": "short paragraph"\n'
                    "}\n\n"
                    "IMPORTANT:\n"
                    "- strength and weakness MUST be arrays (bullet points)\n"
                    "- Do not return paragraphs inside them\n"
                    "- rating must be between 0 to 10"
                ),
            },
            {"role": "user", "content": transcript_text},
        ],
        "temperature": 0.6,
        "max_tokens": 500,
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        ev = json.loads(raw)
        return {
            "strength": ev.get("strength", []),
            "weakness": ev.get("weakness", []),
            "rating": float(ev.get("rating", 0)),
            "summary": ev.get("summary", ""),
        }
    except Exception as exc:
        return {
            "strength": [],
            "weakness": [],
            "rating": 0,
            "summary": f"Could not generate evaluation: {exc}",
        }
