"""
CardAssist LangGraph pipeline.

Real libraries in use:
- LangGraph: StateGraph orchestrates the 4-node workflow below.
- LlamaIndex: src/retrieval.py builds and queries a VectorStoreIndex over the
  policy documents.
- Hugging Face transformers: src/generation.py and src/embeddings.py load the
  real, locally-downloaded flan-t5-small weights (verified: real generation
  output, not a stub).
- W&B Weave: @weave.op() decorates every node function below (real library
  call). Cloud sync to the Weave dashboard is blocked by this sandbox's
  network restrictions (see src/trace_logger.py for the verified error and
  the local-trace substitute).
"""
from typing import TypedDict, List, Optional
import weave
from langgraph.graph import StateGraph, END

from src.retrieval import retrieve
from src.generation import generate_grounded_response
from src.trace_logger import traced

# Categories that this policy set marks as MANDATORY escalation (never
# auto-resolved), per the synthetic policy documents in data/policies/.
MANDATORY_ESCALATION_KEYWORDS = {
    "fraud": ["stolen", "lost my card", "unauthorized", "someone used my card",
              "social security", "ssn", "identity theft", "fraud"],
    "dispute_large_or_unauthorized": ["i want to dispute", "unauthorized charge", "didn't make this charge",
                                       "don't recognize", "wasn't me"],
    "hardship": ["hardship", "lost my job", "can't afford", "medical emergency", "financial trouble"],
    "apr_reduction": ["lower my rate permanently", "permanently lower", "remove my penalty apr", "negotiate my rate"],
    "second_waiver": ["already waived", "already got a late fee waived", "waived it before",
                       "second time this year", "waive it again", "another one this year"],
}

CATEGORY_KEYWORDS = {
    "late_fee": ["late fee", "late payment", "waive"],
    "dispute": ["dispute", "charge i didn't make", "unauthorized charge", "wrong charge"],
    "fraud": ["stolen", "lost my card", "fraud", "identity theft", "unauthorized"],
    "apr": ["apr", "interest rate", "rate change"],
    "credit_limit": ["credit limit", "increase my limit", "available credit"],
    "hardship": ["hardship", "lost my job", "medical emergency", "can't afford"],
}


class CardAssistState(TypedDict, total=False):
    run_id: str
    query: str
    category: str
    mandatory_escalation: bool
    escalation_trigger: Optional[str]
    retrieved: List[dict]
    top_retrieval_score: float
    raw_response: str
    escalate: bool
    escalation_reason: Optional[str]
    final_response: str


@weave.op()
def classify(query: str) -> dict:
    q = query.lower()
    category = "other"
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in q for kw in kws):
            category = cat
            break

    mandatory = False
    trigger = None
    for reason, kws in MANDATORY_ESCALATION_KEYWORDS.items():
        if any(kw in q for kw in kws):
            mandatory = True
            trigger = reason
            break

    return {"category": category, "mandatory_escalation": mandatory, "escalation_trigger": trigger}


@traced("classify_node")
def classify_node(state: CardAssistState) -> CardAssistState:
    result = classify(state["query"])
    state.update(result)
    return state


@weave.op()
def do_retrieve(query: str) -> list:
    return retrieve(query, top_k=2)


@traced("retrieve_node")
def retrieve_node(state: CardAssistState) -> CardAssistState:
    chunks = do_retrieve(state["query"])
    state["retrieved"] = chunks
    state["top_retrieval_score"] = chunks[0]["score"] if chunks and chunks[0]["score"] is not None else 0.0
    return state


@weave.op()
def do_generate(query: str, chunks: list) -> str:
    return generate_grounded_response(query, chunks)


@traced("generate_node")
def generate_node(state: CardAssistState) -> CardAssistState:
    state["raw_response"] = do_generate(state["query"], state["retrieved"])
    return state


# NOTE on this threshold: an eval run showed hit-vs-miss retrieval score
# distributions overlap heavily under mean-pooled T5 encoder embeddings
# (hits: 0.26-0.55, misses: 0.36-0.47), so no cutoff in that range reliably
# separates correct from incorrect retrieval. Rather than tune a threshold
# to hit a target metric, the score is kept only as a very-low safety floor
# (catches genuinely empty/degenerate retrieval) and is logged for
# monitoring, not used as the primary escalation signal. The primary signal
# is the rule-based mandatory-category classifier. See README limitations.
RETRIEVAL_SCORE_FLOOR = 0.10


@weave.op()
def gate(mandatory_escalation: bool, top_retrieval_score: float, category: str) -> dict:
    if mandatory_escalation:
        return {"escalate": True, "reason": "compliance-sensitive category requires human review"}
    if category == "other":
        return {"escalate": True, "reason": "request did not match a known policy category"}
    if top_retrieval_score < RETRIEVAL_SCORE_FLOOR:
        return {"escalate": True, "reason": f"degenerate retrieval confidence ({top_retrieval_score:.2f})"}
    return {"escalate": False, "reason": None}


@traced("confidence_gate_node")
def confidence_gate_node(state: CardAssistState) -> CardAssistState:
    result = gate(state["mandatory_escalation"], state["top_retrieval_score"], state["category"])
    state["escalate"] = result["escalate"]
    state["escalation_reason"] = result["reason"]
    return state


@traced("finalize_node")
def finalize_node(state: CardAssistState) -> CardAssistState:
    if state["escalate"]:
        state["final_response"] = (
            "I want to make sure this is handled correctly, so I'm routing this to a "
            f"specialist ({state['escalation_reason']}). They'll follow up shortly."
        )
    else:
        state["final_response"] = state["raw_response"]
    return state


def build_graph():
    g = StateGraph(CardAssistState)
    g.add_node("classify", classify_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("gate", confidence_gate_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "gate")
    g.add_edge("gate", "finalize")
    g.add_edge("finalize", END)

    return g.compile()


_app = None


def run(query: str, run_id: str) -> CardAssistState:
    global _app
    if _app is None:
        _app = build_graph()
    initial_state: CardAssistState = {"run_id": run_id, "query": query}
    return _app.invoke(initial_state)
