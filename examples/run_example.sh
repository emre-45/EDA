#!/usr/bin/env bash
# Example wrapper to run the EDA script in a venv
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$HOME/.venvs/first-venv/bin/activate" || source .venv/bin/activate
PYTHONPATH=$ROOT python $ROOT/eda_tool/run_eda.py --input $ROOT/eda_tool/tests/fixtures/sample.csv --target target --output $ROOT/sample_eda.html
echo "Report written to $ROOT/sample_eda.html"
open "$ROOT/sample_eda.html"

 PYTHONPATH=$PWD python eda_tool/run_eda.py --input house_prices_train.csv --target SalePrice --output house_prices_eda.html --log-level INFO
 PYTHONPATH=$PWD python eda_tool/run_eda.py --input dermatology_database_1.csv --target class --output dermatology_eda.html --log-level INFO
 PYTHONPATH=$PWD python eda_tool/run_eda.py --input wesad_baseline_vs_stress_wrist.csv --target y --output wesad_eda.html --log-level INFO
PYTHONPATH=$PWD python eda_tool/run_eda.py --input heart.csv --target HeartDisease --output heart_eda.html --log-level INFO
PYTHONPATH=$PWD python eda_tool/run_eda.py --input train.csv --target addicted_label --output phoen_addiction_eda.html --log-level INFO