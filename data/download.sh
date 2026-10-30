#!/usr/bin/env bash
# Banking77 (Casanueva et al., 2020): 13,083 real customer-service queries to a digital bank, 77 intents.
# Source: PolyAI-LDN/task-specific-datasets (CC BY 4.0).
set -euo pipefail
cd "$(dirname "$0")"
COMMIT=57ec275d8078af65b7731c2a98be812d844a6d6b
BASE="https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/$COMMIT/banking_data"
for f in train test; do
  [ -f "$f.csv" ] || curl -fsSL -o "$f.csv" "$BASE/$f.csv"
done
sha256sum -c - <<'EOF'
b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b  train.csv
d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d  test.csv
EOF
