# CardAssist: Auditable Customer-Servicing GenAI System

A LangGraph-orchestrated servicing workflow that classifies customer requests,
retrieves supporting policy text via LlamaIndex, generates source-grounded
responses with a real Hugging Face model, and routes low-confidence or
compliance-sensitive cases to human escalation instead of auto-responding.
Instrumented with W&B Weave (`@weave.op()`) for per-node tracing, evaluated
against a labeled test set for classification accuracy, escalation
correctness, retrieval quality, groundedness, and latency.

## Why this exists

Built to close a specific gap for a GenAI-focused Data Scientist application:
demonstrated hands-on use of LangGraph, LlamaIndex, Hugging Face, and W&B
Weave together in one pipeline, with real weights and a real eval run — not
a description of the tools, an actual working system.

## Architecture

```
query -> classify -> retrieve -> generate -> confidence_gate -> finalize
```

1. **classify** — rule-based classifier assigns a policy category (late fee,
   dispute, fraud, APR, credit limit, hardship, other) and flags
   compliance-sensitive requests that must never be auto-resolved (fraud,
   large/unauthorized disputes, hardship enrollment, permanent rate
   negotiation, repeat fee waivers), based on the synthetic policy set in
   `data/policies/`.
2. **retrieve** — LlamaIndex `VectorStoreIndex` over 6 synthetic servicing
   policy documents, using a custom embedding model (see below).
3. **generate** — `google/flan-t5-small` (real weights, downloaded by the
   user and loaded locally, verified generating genuine grounded answers —
   not a stub) generates a response conditioned only on the retrieved policy
   text.
4. **confidence_gate** — escalates to a human specialist if the request is
   compliance-sensitive, unrecognized, or retrieval returned degenerate
   results. Otherwise returns the generated answer.
5. **finalize** — produces the final customer-facing message.

Every node function is decorated with `@weave.op()` (real `weave` library
usage). See **Sandbox limitations** below for what Weave could and couldn't
do here.

## Real results (n=20 labeled test cases, `data/test_set.jsonl`)

| Metric | Value |
|---|---|
| Category classification accuracy | 90% (18/20) |
| Escalation decision accuracy | 100% (20/20) |
| Escalation recall on mandatory-escalation cases | 100% |
| False-escalation rate (unnecessarily escalated) | 0% |
| Retrieval hit rate (correct policy doc in top-2) | 82.4% (14/17 scoreable cases) |
| Avg. groundedness (lexical overlap, non-escalated responses) | 62.5% |
| Generation latency (mean / p95) | 298ms / 498ms |
| Retrieval latency (mean / p95) | 69ms / 25ms |
| End-to-end latency (mean / p95) | ~370ms / ~530ms |

Full per-case results: `results/eval_results.json`. Full node-level trace
log: `results/trace_log.jsonl`.

## What "auditable" actually means here, with numbers

- Every one of the 8 test cases requiring mandatory human review (fraud,
  large disputes, hardship, permanent rate changes, repeat waivers) was
  correctly escalated — 100% recall on the safety-critical path.
- Zero cases were escalated unnecessarily in the final run.
- Two category-classification misses (a disguised dispute and a disguised
  fraud report) were still correctly escalated anyway, because the gate's
  fail-safe default is to escalate on an unrecognized ("other") category
  rather than guess. Wrong classification did not translate into an unsafe
  auto-response — the system fails toward caution, not silence.

## Honest limitations (found during build, not swept under the rug)

- **Retrieval confidence isn't well-calibrated.** Mean-pooled `flan-t5-small`
  encoder embeddings were used for LlamaIndex retrieval (see design note
  below) because a dedicated sentence-embedding model couldn't be downloaded
  in this sandbox. An initial version gated escalation on an absolute
  cosine-similarity threshold; a real eval run showed correct-retrieval and
  incorrect-retrieval score distributions overlap heavily (hits: 0.26-0.55,
  misses: 0.36-0.47), so no threshold in that range reliably separates them.
  Rather than tune a threshold until the eval numbers looked good, the
  score was demoted to a low safety-floor check and logged for monitoring;
  the primary escalation signal is the rule-based mandatory-category
  classifier. Production fix: a proper sentence-similarity-trained
  embedding model (e.g. `sentence-transformers/all-MiniLM-L6-v2`).
- **Rule-based classifier has a real recall ceiling.** Two category misses
  in the 20-case set were paraphrases the keyword list didn't cover
  ("I don't recognize this charge" vs. the trigger word "dispute";
  "used my card without permission" vs. "unauthorized"). Caught safely by
  the fail-to-escalate default, but a production version would use the LLM
  itself (or a fine-tuned classifier) for intent classification instead of
  keyword matching.
- **Small test set (n=20).** Enough to surface and fix real bugs (a
  mis-scoped escalation trigger, a miscalibrated threshold) but not a
  statistically powered evaluation. Would need a larger, ideally
  human-labeled set for a production accuracy claim.
- **Policy documents are synthetic**, authored for this project (labeled
  "TestCard Servicing Guidelines, Synthetic" in each file) — not real
  Capital One or any other company's policy text.

## Design note: why the embedding model reuses flan-t5-small

LlamaIndex's default embedding backends assume network access to OpenAI or
the Hugging Face Hub for a dedicated embedding model. This sandbox has
neither (`huggingface.co` and `api.wandb.ai` both return 403 from the egress
proxy — verified, not assumed). Downloading a second model was not an
option, so `src/embeddings.py` wraps the already-verified real
`flan-t5-small` encoder (mean-pooled last hidden state) to produce genuine
model-derived embeddings instead of fabricating a numeric stand-in. This is
a legitimate technique, but not what a production system would use (see
limitations above).

## Sandbox limitations for W&B Weave

`weave.init()` requires a live connection to `api.wandb.ai`; this sandbox's
egress proxy returns 403 for that domain (verified directly, not assumed).
`@weave.op()` decorators work standalone without `.init()` — confirmed: the
decorated functions run normally and Weave prints a warning that traces
won't be logged remotely, rather than failing. This project uses real
`@weave.op()` decorators on every pipeline node (`src/graph.py`) and adds a
local JSONL trace logger (`src/trace_logger.py`) that captures the same
per-call data (inputs, outputs, latency, call id) Weave would have sent to
its dashboard, so the eval report has real trace data to work from. In an
environment with network access, the existing `@weave.op()` calls would
start logging to the Weave dashboard with no code changes — only
`weave.init("cardassist")` needs to be added back.

## Repo layout

```
CardAssist/
├── README.md
├── data/
│   ├── policies/            6 synthetic servicing policy documents
│   └── test_set.jsonl       20 labeled test queries
├── src/
│   ├── embeddings.py        T5-encoder embedding wrapper for LlamaIndex
│   ├── retrieval.py         LlamaIndex VectorStoreIndex + retrieval
│   ├── generation.py        flan-t5-small grounded generation
│   ├── graph.py             LangGraph pipeline + Weave instrumentation
│   ├── trace_logger.py      Local trace capture (Weave cloud substitute)
│   └── eval.py               Eval harness — run this to reproduce results
└── results/
    ├── eval_results.json    Full per-case results + summary metrics
    └── trace_log.jsonl      Per-node trace log (100 records, 20 runs × 5 nodes)
```

## Reproducing

```bash
pip install langgraph llama-index-core transformers sentencepiece torch weave
# place the flan-t5-small model weights at ./models/flan-t5-small/flan-t5-small/
python3 -m src.eval
```
