# CardAssist and archived CallAssist evidence

This repository contains two distinct portfolio demonstrations. Their datasets,
metrics and deployment status must not be combined.

| Component | Entry point | Evidence and status |
|---|---|---|
| CardAssist text RAG | `python -m src.eval` | 20 synthetic labeled queries in `data/test_set.jsonl`; fresh results in `evidence/repair_2026-10-19/eval_results.json` |
| CallAssist audio API | `legacy/callassist/run_live_eval.py` | 15 saved historical API responses (13/15 correct classification) in `legacy/callassist/live_eval_results.json`; not a verification of a currently running service |

## Run CardAssist

Use Python 3.11 or 3.12 in a virtual environment, then:

```bash
python -m pip install -r requirements.txt
python -m src.eval
```

The default model is `google/flan-t5-small`; the first run downloads its weights.
For offline execution set `CARDASSIST_MODEL_PATH` to a local model directory
containing its tokenizer and weights. Both retrieval embeddings and generation
use that same setting. Results and local traces are written to `results/`.

The workflow uses LangGraph, LlamaIndex, Hugging Face transformers and Weave
operation decorators. Weave cloud upload is not configured; the JSONL logger
captures local node inputs, changed outputs and latency. Use only synthetic or
appropriately sanitized inputs: local traces include input text.

Policies in `data/policies/` are synthetic. Classification and mandatory review
rules are keyword-based. T5 encoder embeddings are not a dedicated sentence
similarity model. The 20 cases are a development fixture, not an independent
production benchmark. Lexical overlap is a limited groundedness proxy.

The repaired checkout was evaluated on all 20 development cases: 18/20 category
predictions correct, 20/20 escalation decisions correct, and 14/17 scoreable
retrievals successful. Full per-case evidence and 100 node traces are committed
under `evidence/repair_2026-10-19/`. These small development-set results do not
establish production reliability. Response quality remains limited: lexical
overlap is not a factual correctness or usefulness score.

## CallAssist historical audio evaluation

Original project ZIPs are retained as historical source archives. Audio fixtures,
the original labeled cases, saved API responses, and the evaluation client are
under `legacy/callassist/`. Set `CALLASSIST_BASE_URL` to your own HTTPS deployment
and `CALLASSIST_API_KEY` to a newly rotated credential before invoking the client.
It writes `live_eval_results_new.json`, preserving the original evidence.

A hardcoded credential was removed from the client. It remains in Git history
and must be revoked/rotated by the deployment owner. This change does not rotate
it or rewrite history. The old endpoint has not been contacted during repair.

## Verification

```bash
python -m pip install pytest
python -m pytest tests -q
```

The trace regression test verifies that in-place node updates preserve before
and after values. It does not load model weights or validate end-to-end accuracy.
See `VALIDATION.md` for the checks actually run during repair.
