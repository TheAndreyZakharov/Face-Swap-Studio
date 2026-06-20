#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_DIR/data/temp/face-swap-studio.pid"
LOG_FILE="$PROJECT_DIR/data/temp/face-swap-studio.log"

cd "$PROJECT_DIR"
mkdir -p data/temp

if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"

    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "Face Swap Studio уже запущен. PID: $OLD_PID"
        echo "Адрес: http://127.0.0.1:7860"
        exit 0
    fi

    rm -f "$PID_FILE"
fi

source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

nohup python app.py > "$LOG_FILE" 2>&1 &
APP_PID=$!
echo "$APP_PID" > "$PID_FILE"

sleep 3

if kill -0 "$APP_PID" 2>/dev/null; then
    echo "Face Swap Studio запущен."
    echo "PID: $APP_PID"
    echo "Адрес: http://127.0.0.1:7860"
    echo "Лог: $LOG_FILE"
    open "http://127.0.0.1:7860"
else
    echo "Приложение не запустилось."
    echo "Последние строки лога:"
    tail -n 50 "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi