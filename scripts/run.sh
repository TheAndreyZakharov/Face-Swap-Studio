#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
    echo "Не найдено основное окружение: $PROJECT_ROOT/.venv"
    exit 1
fi

source ".venv/bin/activate"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false

exec python app.py