"""FastAPI service: POST /route {"text": "..."} -> team, action, probabilities."""
from __future__ import annotations

import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import apply_policy, team_probs

MODEL = Path(os.environ.get("CARDROUTE_MODEL", Path(__file__).resolve().parents[2] / "results" / "model.joblib"))
app = FastAPI(title="CardRoute")
_bundle: dict = {}


class Message(BaseModel):
    text: str = Field(min_length=2, max_length=2000)


def _load():
    if not _bundle:
        if not MODEL.exists():
            raise HTTPException(503, "model not trained: run python -m cardroute.model")
        _bundle.update(joblib.load(MODEL))
    return _bundle


@app.get("/health")
def health():
    return {"status": "ok", "model_present": MODEL.exists()}


@app.post("/route")
def route(msg: Message):
    b = _load()
    tp = team_probs(b["model"], [msg.text])
    pol = apply_policy(tp, b["t_urgent"], b["t_auto"]).iloc[0]
    intent = b["model"].predict([msg.text])[0]
    return {"action": pol["action"], "team": "Fraud & security" if pol["action"] == "escalate_fraud" else pol["predicted_team"],
            "intent": intent, "confidence": round(float(pol["confidence"]), 4),
            "p_fraud": round(float(pol["p_fraud"]), 4)}
