from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cardroute.model import apply_policy, min_threshold_for, urgent_threshold
from cardroute.teams import INTENT_TO_TEAM, TEAMS, URGENT_TEAM

DATA = Path(__file__).parents[1] / "data" / "train.csv"
MODEL = Path(__file__).parents[1] / "results" / "model.joblib"


def test_every_intent_maps_to_exactly_one_team():
    flat = [i for intents in TEAMS.values() for i in intents]
    assert len(flat) == len(set(flat)) == 77
    if DATA.exists():
        assert set(pd.read_csv(DATA)["category"]) == set(INTENT_TO_TEAM)


def test_thresholds():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    ok = np.array([1, 1, 1, 0, 0], dtype=bool)
    assert min_threshold_for(scores, ok, 0.99) == pytest.approx(0.7)
    p_urgent = np.array([0.9, 0.5, 0.2, 0.05])
    is_urgent = np.array([True, True, True, False])
    assert urgent_threshold(p_urgent, is_urgent, 1.0) == pytest.approx(0.2)


def test_policy_escalates_before_auto_routing():
    tp = pd.DataFrame([{t: 0.0 for t in TEAMS} | {"Cards": 0.7, URGENT_TEAM: 0.3},
                       {t: 0.0 for t in TEAMS} | {"Cards": 0.95, URGENT_TEAM: 0.05},
                       {t: 0.0 for t in TEAMS} | {"Cards": 0.4, "Top-ups": 0.35, URGENT_TEAM: 0.05}])
    pol = apply_policy(tp, t_urgent=0.2, t_auto=0.6)
    assert pol["action"].tolist() == ["escalate_fraud", "auto_route", "human_triage"]


@pytest.mark.skipif(not MODEL.exists(), reason="train first: python -m cardroute.model")
def test_api_routes_real_style_messages():
    from fastapi.testclient import TestClient
    from cardroute.api import app
    c = TestClient(app)
    stolen = c.post("/route", json={"text": "someone stole my card and is using it, please block it"}).json()
    assert stolen["action"] == "escalate_fraud"
    topup = c.post("/route", json={"text": "my top-up failed, why did that happen?"}).json()
    assert topup["team"] == "Top-ups"
    assert c.post("/route", json={"text": ""}).status_code == 422
