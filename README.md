# CardRoute: Routing Bank Customer Messages, With a Fraud Safety Net

**The problem.** A digital bank's support inbox gets thousands of free-text messages a day: "my card hasn't
arrived", "I don't recognise this payment", "why was I charged for this top-up?". Each must reach the right
team. A misrouted fee question costs a few minutes; a misrouted "someone is using my card" costs the
customer money. The bank needs automatic routing that is accurate, and that almost never lets a fraud
message slip into a slow queue.

**Data:** [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (Casanueva et al., 2020): 13,083
real customer queries to a digital bank, labeled with 77 intents. `data/download.sh` pins the source commit
and checks SHA-256 hashes. The 77 intents are grouped into 6 support teams in `src/cardroute/teams.py`
(this grouping is the project's design, written out so it can be reviewed).
**Stack:** Python, scikit-learn, pandas, FastAPI, pytest, GitHub Actions.

## How it works

1. **Classifier:** TF-IDF on words and character n-grams, logistic regression over the 77 intents. Team
   probability is the sum of its intents' probabilities. Chosen on a validation split (15% of training data).
2. **Routing policy**, with thresholds set on validation and the test set scored once:
   - **Fraud safety net:** if P(Fraud & security) ≥ 0.160, escalate to the fraud queue immediately.
     The threshold was set to catch at least 98% of fraud messages on validation.
   - **Auto-route:** otherwise, if the top team's probability is high enough, route automatically.
   - **Human triage:** everything else.

## Results on the held-out test set (3,080 messages, `results/results.json`)

| | Result |
|---|---|
| Intent accuracy (77 classes) | 91.0% (macro F1 0.911) |
| Team routing accuracy (6 teams) | 97.4% |
| **Fraud messages caught by the safety net** | **315 of 320 (98.4%)** |
| Escalations that really were fraud | 89.0% |
| Auto-routed, at the 97% validation target | 88.5% of messages, 98.0% correct |

Model selection on validation: word TF-IDF alone scored 88.5% intent accuracy; adding character n-grams raised
it to 91.5%, so the word + character model was used.

**Choosing how much to automate.** A stricter auto-route target sends more messages to people:

| Validation accuracy target | Auto-routed (test) | Accuracy of auto-routed | Sent to human triage |
|---|---|---|---|
| 97% | 88.5% | 98.0% | 0.03% |
| 98% | 87.2% | 98.5% | 1.3% |
| 99% | 82.1% | 99.6% | 6.5% |
| 99.5% | 76.4% | 99.8% | 12.1% |

(The remaining 11.5% of messages are fraud escalations in every row.) At the 99% target, 1 in 15 messages
needs a person and auto-routed messages are wrong about 1 time in 280.

**What it still gets wrong.** The 5 missed fraud messages include "I cannot find my credit card" and three ATM
card-swallowed messages written in unusual words ("I was taking out funds and was unable to regain my card").
The most common intent mix-ups are near-duplicates that land on the same team, such as top-up failed vs top-up
reverted and card arrival vs card delivery estimate. Per-team routing accuracy ranges from 94.4% (Fraud &
security) to 99.7% (Account & identity).

## Run

```bash
pip install -e ".[test]"
bash data/download.sh
python -m cardroute.model          # about a minute on CPU; writes results/results.json and model.joblib
pytest -q                          # 4 tests, including API calls
uvicorn cardroute.api:app          # POST /route {"text": "I lost my card"}
```

## Limitations

- Banking77 messages are single-intent and fairly short; a real inbox has multi-issue messages and threads.
- The team grouping is a design choice. A bank would set it to match its own queues and retrain thresholds.
- Probabilities come from the classifier directly; recalibrate thresholds whenever the message mix changes.

## Earlier work in this repo

`legacy/` keeps the earlier CardAssist RAG demo (20 synthetic test queries, synthetic policies) and the
CallAssist audio API evidence. **Security note:** an API credential was committed in this repo's history by an
earlier version. It must be revoked by the deployment owner; removing it from the current files doesn't remove
it from history.
