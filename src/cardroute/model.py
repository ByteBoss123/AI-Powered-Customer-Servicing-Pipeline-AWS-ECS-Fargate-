"""Train the message router, set its thresholds on validation data, and score the test set once.

Routing policy applied to every incoming message:
1. Fraud safety net: if P(Fraud & security) >= the escalation threshold, send to the fraud queue now.
   Threshold chosen on validation to catch at least 98% of fraud/security messages.
2. Otherwise, if the top team's probability >= the auto-route threshold, route automatically.
   Threshold chosen on validation so auto-routed messages are at least 97% correct.
3. Everything else goes to a human triage queue.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from .teams import INTENT_TO_TEAM, TEAMS, URGENT_TEAM

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SEED = 42
URGENT_RECALL_TARGET = 0.98
AUTO_ROUTE_ACCURACY_TARGET = 0.97


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{split}.csv")
    df["team"] = df["category"].map(INTENT_TO_TEAM)
    if df["team"].isna().any():
        raise ValueError(f"unmapped intents: {sorted(df.loc[df.team.isna(), 'category'].unique())}")
    return df


def make_model(char: bool, C: float) -> Pipeline:
    feats = [("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1))]
    if char:
        feats.append(("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)))
    return Pipeline([("feats", FeatureUnion(feats)),
                     ("clf", LogisticRegression(C=C, max_iter=3000))])


def team_probs(model: Pipeline, texts) -> pd.DataFrame:
    p = pd.DataFrame(model.predict_proba(texts), columns=model.classes_)
    return p.T.groupby(lambda intent: INTENT_TO_TEAM[intent]).sum().T[list(TEAMS)]


def min_threshold_for(scores: np.ndarray, ok: np.ndarray, target: float) -> float:
    """Lowest threshold whose selected set meets the accuracy target (maximizes coverage)."""
    for t in np.unique(np.round(scores, 4)):
        sel = scores >= t
        if sel.any() and ok[sel].mean() >= target:
            return float(t)
    return 1.0


def urgent_threshold(p_urgent: np.ndarray, is_urgent: np.ndarray, target: float) -> float:
    """Highest threshold that still catches `target` of urgent messages."""
    cands = np.sort(np.unique(np.round(p_urgent[is_urgent], 4)))[::-1]
    for t in cands:
        if (p_urgent[is_urgent] >= t).mean() >= target:
            return float(t)
    return 0.0


def apply_policy(tp: pd.DataFrame, t_urgent: float, t_auto: float) -> pd.DataFrame:
    top_team = tp.idxmax(axis=1)
    top_p = tp.max(axis=1)
    action = np.where(tp[URGENT_TEAM] >= t_urgent, "escalate_fraud",
                      np.where(top_p >= t_auto, "auto_route", "human_triage"))
    return pd.DataFrame({"predicted_team": top_team, "confidence": top_p,
                         "p_fraud": tp[URGENT_TEAM], "action": action})


def run(out: Path = ROOT / "results") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    train_all, test = load("train"), load("test")
    fit, val = train_test_split(train_all, test_size=0.15, stratify=train_all["category"], random_state=SEED)

    candidates = {"word tf-idf, C=10": (False, 10.0), "word + char tf-idf, C=10": (True, 10.0),
                  "word + char tf-idf, C=30": (True, 30.0)}
    val_scores, models = {}, {}
    for name, (char, C) in candidates.items():
        m = make_model(char, C).fit(fit["text"], fit["category"])
        pred = m.predict(val["text"])
        val_scores[name] = {"accuracy": round(accuracy_score(val["category"], pred), 4),
                            "macro_f1": round(f1_score(val["category"], pred, average="macro"), 4)}
        models[name] = m
    best = max(val_scores, key=lambda k: val_scores[k]["macro_f1"])
    model = models[best]

    # Thresholds from validation only
    tp_val = team_probs(model, val["text"])
    t_urgent = urgent_threshold(tp_val[URGENT_TEAM].to_numpy(), (val["team"] == URGENT_TEAM).to_numpy(),
                                URGENT_RECALL_TARGET)
    non_urgent_val = tp_val[URGENT_TEAM] < t_urgent
    ok_val = (tp_val.idxmax(axis=1) == val["team"].to_numpy())
    t_auto = min_threshold_for(tp_val.max(axis=1)[non_urgent_val].to_numpy(), ok_val[non_urgent_val].to_numpy(),
                               AUTO_ROUTE_ACCURACY_TARGET)

    # Test set, scored once
    pred_intent = model.predict(test["text"])
    tp = team_probs(model, test["text"])
    pol = apply_policy(tp, t_urgent, t_auto)
    true_team = test["team"].to_numpy()
    is_urgent = true_team == URGENT_TEAM
    esc = (pol["action"] == "escalate_fraud").to_numpy()
    auto = (pol["action"] == "auto_route").to_numpy()
    team_ok = pol["predicted_team"].to_numpy() == true_team

    # Trade-off: stricter auto-route targets send more messages to people (thresholds set on validation)
    tradeoff = []
    for target in (0.97, 0.98, 0.99, 0.995):
        t = min_threshold_for(tp_val.max(axis=1)[non_urgent_val].to_numpy(), ok_val[non_urgent_val].to_numpy(), target)
        a = (~esc) & (pol["confidence"].to_numpy() >= t)
        tradeoff.append({"validation_target": target, "threshold": round(t, 4),
                         "test_auto_routed_share": round(float(a.mean()), 4),
                         "test_auto_routed_accuracy": round(float(team_ok[a].mean()), 4),
                         "test_to_human_share": round(float(((~esc) & ~a).mean()), 4)})

    confusions = Counter((t, p) for t, p in zip(test["category"], pred_intent) if t != p).most_common(8)
    per_team = {team: round(float(team_ok[true_team == team].mean()), 4) for team in TEAMS}

    report = {
        "data": {"source": "Banking77 (Casanueva et al., 2020), real customer queries to a digital bank",
                 "intents": 77, "teams": len(TEAMS), "train_fit": len(fit), "validation": len(val), "test": len(test),
                 "test_messages_by_team": {k: int(v) for k, v in test["team"].value_counts().items()}},
        "model_selection_validation": val_scores,
        "chosen_model": best,
        "test_intent": {"accuracy": round(accuracy_score(test["category"], pred_intent), 4),
                        "macro_f1": round(f1_score(test["category"], pred_intent, average="macro"), 4)},
        "test_team_routing_accuracy": round(float(team_ok.mean()), 4),
        "test_team_routing_accuracy_by_team": per_team,
        "policy": {
            "urgent_escalation_threshold": round(t_urgent, 4),
            "auto_route_threshold": round(t_auto, 4),
            "test_fraud_messages": int(is_urgent.sum()),
            "test_fraud_caught_by_escalation": int((esc & is_urgent).sum()),
            "test_fraud_recall": round(float(esc[is_urgent].mean()), 4),
            "test_escalations_that_were_fraud": round(float(is_urgent[esc].mean()), 4),
            "test_fraud_missed_examples": test.loc[is_urgent & ~esc, ["text", "category"]].head(5).to_dict("records"),
            "test_auto_routed_share": round(float(auto.mean()), 4),
            "test_auto_routed_accuracy": round(float(team_ok[auto].mean()), 4),
            "test_human_triage_share": round(float((pol["action"] == "human_triage").mean()), 4),
            "test_escalated_share": round(float(esc.mean()), 4),
        },
        "auto_route_tradeoff": tradeoff,
        "test_top_intent_confusions": [{"true": t, "predicted": p, "count": c} for (t, p), c in confusions],
    }
    (out / "results.json").write_text(json.dumps(report, indent=2))
    joblib.dump({"model": model, "t_urgent": t_urgent, "t_auto": t_auto}, out / "model.joblib")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))
