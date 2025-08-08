alias l := lint

# run once to initialize env
init:
    uv sync
    uv run pytest -p pytest_pretty

# run project formatters/linters
lint:
    uv run ruff check --fix
    uv run ruff format
    just --fmt --unstable
    markdownlint-cli2 . --fix
    uv run mypy

# run ruff with unsafe-fixes
ruff_unsafe:
    uv run ruff check --fix --unsafe-fixes

# Detect and display canny edges on a set of test images. Use ctrl+c to kill
test:
    uv run pytest
