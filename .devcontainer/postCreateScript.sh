#!/bin/bash

# # install astral-uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# PATH="/home/ubuntu/.local/bin:$PATH"

# Initialize repository as an astral-uv project if not already initialized
if [ ! -f "./pyproject.toml" ]; then
    uv init
fi

# Create an astral-uv virtual environment if the virtual environment does not exist
if [ ! -d "./.venv" ]; then
    uv venv --python 3.12
fi

# Activate the virtual environment and install dependencies
source ./.venv/bin/activate
uv sync
uv add torch numpy pandas matplotlib jupyterlab

# add astral-uv venv activation to bashrc
echo 'source ./.venv/bin/activate' >> ~/.bashrc

# check nvidia-smi
echo $(nvidia-smi) > nvidia-smi.out.txt