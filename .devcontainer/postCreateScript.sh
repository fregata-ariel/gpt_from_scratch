#!/bin/bash

# Create an astral-uv virtual environment if the virtual environment does not exist
if [ ! -d "/workspace/.venv" ]; then
    uv venv --python 3.12
fi

# Activate the virtual environment and install dependencies
source /workspace/.venv/bin/activate
uv sync
uv add pytorch