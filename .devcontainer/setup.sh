#!/bin/bash

set -e

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="${HOME}/.local/bin:${PATH}"

uv sync --extra test --extra plotting --group dev
uv run pre-commit install
