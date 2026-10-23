# Automated EDA Tool

Quick start for the EDA tool implemented under `eda_tool/`.

Prerequisites
- Python 3.12+
- A virtual environment (recommended)

Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r eda_tool/requirements.txt
```

Run the sample
```bash
# from workspace root
PYTHONPATH=$PWD python eda_tool/run_eda.py --input eda_tool/tests/fixtures/sample.csv --target target --output sample_eda.html
open sample_eda.html
```

Entry point: `eda_tool/run_eda.py` — see that file for CLI options.

Project layout
- `eda_tool/eda_core/` — core modules (type inference, plotting, stats, report)
- `eda_tool/templates/` — Jinja2 HTML template
- `eda_tool/tests/` — unit tests and sample fixture
