#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$PROJECT_DIR/data/temp/face-swap-studio.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "PID-файл отсутствует. Приложение, вероятно, не запущено в фоне."
    exit 0
fi

APP_PID="$(cat "$PID_FILE")"

if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "Процесс $APP_PID уже не работает."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Останавливаю Face Swap Studio, PID: $APP_PID"
kill "$APP_PID"

for _ in {1..20}; do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "Face Swap Studio остановлен."
        exit 0
    fi

    sleep 0.25
done

echo "Обычная остановка не сработала. Завершаю процесс принудительно."
kill -9 "$APP_PID" 2>/dev/null || true
rm -f "$PID_FILE"
echo "Face Swap Studio остановлен."