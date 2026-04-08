#!/bin/bash
set -e

echo "[start.sh] Starting FastAPI on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 &

echo "[start.sh] Starting Gradio on port 7861..."
python app/gradio_demo.py &

echo "[start.sh] Starting nginx on port 7860"
sleep 3
nginx -g "daemon off;"