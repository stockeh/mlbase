# Base Machine Learning Code

## Local Usage
This project is now managed with [uv](https://github.com/astral-sh/uv) so everything lives in a
standard `pyproject.toml`.

```bash
# create the environment and install the core package + dev tools
uv sync

# run any command inside the environment
uv run python -c "import mlbase"
```

### Common Tasks
The repo uses Astral tooling by default:

- `uv run ty lint` &mdash; run Ruff checks
- `uv run ty fmt` &mdash; apply Ruff formatting
- `uv run ty tests` &mdash; execute the pytest suite

The package only declares its core runtime dependency (`numpy`) and a single dev group (ty, Ruff,
pytest). Install other frameworks (PyTorch, TensorFlow, etc.) in your virtual environment as needed
with `uv add` so they stay opt-in.  
