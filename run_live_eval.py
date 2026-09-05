"""
Run this from your machine (needs `requests` — pip install requests) to hit
the LIVE CallAssist AWS deployment with real synthesized audio and get
genuine, verifiable numbers: transcription accuracy (word error rate),
call-reason classification accuracy, and end-to-end latency.

Usage:
    pip install requests
    python run_live_eval.py
"""
import json
import time
import requests

BASE_URL = "http://54.208.48.196:8000"
API_KEY = "cak_df9ea63aeb5944c385deed6553c0d7872b0fd603f00a4ea6"

with open("test_cases.json") as f:
    CASES = json.load(f)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Standard WER via Levenshtein distance on word sequences."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def main():
    results = []
    for case in CASES:
        wav_path = f"wavs/{case['id']}.wav"
        start = time.perf_counter()
        with open(wav_path, "rb") as f:
            resp = requests.post(
                f"{BASE_URL}/v1/calls/audio",
                headers={"X-API-Key": API_KEY},
                files={"file": (f"{case['id']}.wav", f, "audio/wav")},
                params={"call_id": case["id"]},
                timeout=60,
            )
        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            results.append({**case, "error": f"HTTP {resp.status_code}: {resp.text}"})
            continue

        data = resp.json()
        transcript = data.get("redacted_transcript", "")
        predicted_reason = data.get("classification", {}).get("call_reason", "")
        escalated = data.get("escalated")
        wer = word_error_rate(case["text"], transcript)

        results.append({
            "id": case["id"],
            "expected_reason": case["expected_reason"],
            "predicted_reason": predicted_reason,
            "correct": predicted_reason == case["expected_reason"],
            "escalated": escalated,
            "reference_text": case["text"],
            "redacted_transcript": transcript,
            "wer_vs_redacted_transcript": round(wer, 4),
            "latency_ms": round(latency_ms, 1),
            "raw_response": data,
        })

    with open("live_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    if valid:
        accuracy = sum(r["correct"] for r in valid) / len(valid)
        mean_wer = sum(r["wer_vs_redacted_transcript"] for r in valid) / len(valid)
        mean_latency = sum(r["latency_ms"] for r in valid) / len(valid)
        escalation_rate = sum(bool(r.get("escalated")) for r in valid) / len(valid)
        print(f"\n=== LIVE EVAL SUMMARY ({len(valid)}/{len(CASES)} succeeded) ===")
        print(f"Classification accuracy: {accuracy:.1%}")
        print(f"Mean WER (vs PII-redacted transcript, the only transcript the API exposes): {mean_wer:.1%}")
        print(f"Escalation rate: {escalation_rate:.1%}")
        print(f"Mean latency: {mean_latency:.0f}ms")
    if errors:
        print(f"\n{len(errors)} requests failed:")
        for e in errors:
            print(f"  {e['id']}: {e['error']}")

    print("\nFull results written to live_eval_results.json — paste that file's contents back.")


if __name__ == "__main__":
    main()
