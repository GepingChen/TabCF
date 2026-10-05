# Contributing

Issues and pull requests are welcome.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements-core.txt
pip install -r environment/requirements-baselines.txt
```

## Running tests

```bash
pytest tabcf_core interv_mean interv_qtl multivar empirical
```

## Pull requests

- Keep changes focused and include or update tests for any behavior change.
- Make sure `pytest` passes before opening a PR.
- Describe what changed and why in the PR description.
