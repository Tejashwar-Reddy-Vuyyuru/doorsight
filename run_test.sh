#!/usr/bin/env bash
# One-command first test: ./run_test.sh houses.csv
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate
pip install -q -r requirements.txt
if [ -z "$MAPILLARY_TOKEN" ]; then echo "Set MAPILLARY_TOKEN first (free at mapillary.com/developer):  export MAPILLARY_TOKEN=..."; exit 1; fi
python doorsight.py batch "${1:-houses.csv}"
echo; echo "Now:  python app.py   -> http://localhost:8000"
