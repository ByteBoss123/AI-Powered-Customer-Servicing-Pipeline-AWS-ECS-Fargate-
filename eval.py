import json
import os
import uuid
import statistics
from src.graph import run

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "test_set.jsonl")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "eval_results.json")
TRACE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "trace_log.jsonl")

CATEGORY_TO_FILE = {
    "late_fee": "late_fee_policy.txt",
    "dispute": "dispute_policy.txt",
    "fraud": "fraud_policy.txt",
    "apr": "apr_policy.txt",
    "credit_limit": "credit_limit_policy.txt",
    "hardship": "hardship_policy.txt",
}


def lexical_groundedness(response: str, chunks: list) -> float:
    """Fraction of the response's content words that also appear in the
    retrieved policy text. A simple, transparent proxy for groundedness
    (not an LLM-judge score) given no network access to a judge model."""
    resp_words = set(w.strip(".,?!").lower() for w in response.split() if len(w) > 3)
    if not resp_words:
        return 0.0
    context_words = set()
    for c in chunks:
        context_words.update(w.strip(".,?!").lower() for w in c["text"].split() if len(w) > 3)
    if not context_words:
        return 0.0
    overlap = resp_words & context_words
    return len(overlap) / len(resp_words)


def main():
    # fresh trace log for this run
    if os.path.exists(TRACE_PATH):
        os.remove(TRACE_PATH)

    with open(TEST_SET_PATH) as f:
        test_cases = [json.loads(line) for line in f]

    records = []
    for case in test_cases:
        run_id = str(uuid.uuid4())
        result = run(case["query"], run_id)

        pred_category = result.get("category")
        pred_escalate = result.get("escalate")
        retrieved = result.get("retrieved", [])
        retrieved_files = [c["file_name"] for c in retrieved]
        expected_file = CATEGORY_TO_FILE.get(case["gold_category"])
        retrieval_hit = expected_file in retrieved_files if expected_file else None

        groundedness = None
        if not pred_escalate:
            groundedness = lexical_groundedness(result.get("raw_response", ""), retrieved)

        records.append(
            {
                "query": case["query"],
                "gold_category": case["gold_category"],
                "pred_category": pred_category,
                "category_correct": pred_category == case["gold_category"],
                "gold_escalate": case["gold_escalate"],
                "pred_escalate": pred_escalate,
                "escalation_correct": pred_escalate == case["gold_escalate"],
                "retrieval_hit": retrieval_hit,
                "top_retrieval_score": result.get("top_retrieval_score"),
                "groundedness": groundedness,
                "final_response": result.get("final_response"),
            }
        )

    n = len(records)
    category_acc = sum(r["category_correct"] for r in records) / n
    escalation_acc = sum(r["escalation_correct"] for r in records) / n

    # escalation recall: of cases that SHOULD escalate, how many did we catch?
    should_escalate = [r for r in records if r["gold_escalate"]]
    escalation_recall = (
        sum(r["pred_escalate"] for r in should_escalate) / len(should_escalate)
        if should_escalate else None
    )

    # false-escalation rate: of cases that should NOT escalate, how many did we wrongly escalate?
    should_not_escalate = [r for r in records if not r["gold_escalate"]]
    false_escalation_rate = (
        sum(r["pred_escalate"] for r in should_not_escalate) / len(should_not_escalate)
        if should_not_escalate else None
    )

    retrieval_hits = [r["retrieval_hit"] for r in records if r["retrieval_hit"] is not None]
    retrieval_hit_rate = sum(retrieval_hits) / len(retrieval_hits) if retrieval_hits else None

    grounded_scores = [r["groundedness"] for r in records if r["groundedness"] is not None]
    avg_groundedness = statistics.mean(grounded_scores) if grounded_scores else None

    # latency from trace log
    with open(TRACE_PATH) as f:
        traces = [json.loads(line) for line in f]
    latencies_by_node = {}
    for t in traces:
        latencies_by_node.setdefault(t["node"], []).append(t["latency_ms"])
    latency_summary = {
        node: {
            "mean_ms": round(statistics.mean(vals), 1),
            "p95_ms": round(sorted(vals)[int(len(vals) * 0.95) - 1], 1) if len(vals) > 1 else round(vals[0], 1),
        }
        for node, vals in latencies_by_node.items()
    }
    total_latency_per_run = {}
    for t in traces:
        total_latency_per_run.setdefault(t["run_id"], 0)
        total_latency_per_run[t["run_id"]] += t["latency_ms"]
    e2e_latencies = list(total_latency_per_run.values())

    summary = {
        "n_test_cases": n,
        "category_accuracy": round(category_acc, 3),
        "escalation_accuracy": round(escalation_acc, 3),
        "escalation_recall_on_mandatory_cases": round(escalation_recall, 3) if escalation_recall is not None else None,
        "false_escalation_rate": round(false_escalation_rate, 3) if false_escalation_rate is not None else None,
        "retrieval_hit_rate": round(retrieval_hit_rate, 3) if retrieval_hit_rate is not None else None,
        "avg_groundedness_lexical_overlap": round(avg_groundedness, 3) if avg_groundedness is not None else None,
        "latency_by_node_ms": latency_summary,
        "e2e_latency_ms": {
            "mean": round(statistics.mean(e2e_latencies), 1),
            "p95": round(sorted(e2e_latencies)[int(len(e2e_latencies) * 0.95) - 1], 1) if len(e2e_latencies) > 1 else round(e2e_latencies[0], 1),
        },
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)

    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
