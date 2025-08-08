alias l := lint

# run project formatters/linters
lint:
    uv run ruff check --fix
    uv run ruff format
    just --fmt --unstable
    yamllint .
    markdownlint-cli2 . --fix
    uv run mypy

# run ruff with unsafe-fixes
unsafe:
    uv run ruff check --fix --unsafe-fixes

# Show canny detections on a set of test images. Kill with ctrl+c
demo:
    uv run packages/canny_chenn/test/test_demo.py

# run basic tests
test:
    pytest
