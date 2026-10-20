# Repair validation — 2026-10-19

- Trace mutation regression: 1 test passed.
- Full CardAssist evaluation completed with real flan-t5-small weights.
- 20 cases: classification 18/20; escalation decision 20/20; retrieval 14/17.
- Mandatory escalation recall 1.0; false escalation rate 0.0 on this fixture only.
- Mean end-to-end node latency 3594.8 ms; nearest-rank p95 11052.6 ms.
  Includes cold initialization; hardware-dependent, not an SLA.
- Saved 100 node traces and all case-level responses under `evidence/repair_2026-10-19/`.
- Historical CallAssist results are separate: 13 correct classifications / 15.
- No remote API contacted, no cloud deployment, no Weave cloud sync.

The generated answers include terse or inadequate responses despite correct
routing. This run verifies execution and reported metrics, not production
answer quality. No new real-user usage is claimed. The publicly committed
legacy API credential must be revoked/rotated; deleting it from current source
does not remove it from prior commits.
