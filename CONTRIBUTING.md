# Contributing

## Development

Use Python 3.11 for local checks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q data src
ruff check data src
black --check data src
pytest -q
```

Changes affecting training behavior should include a regression test or a reproducible explanation of why one is not practical.

Keep credentials, datasets, checkpoints, and cloud state out of commits.
