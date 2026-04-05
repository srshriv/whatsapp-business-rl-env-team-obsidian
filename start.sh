#!/bin/bash
set -e

# Start Gradio on internal port 7861 (no code change)
python app/gradio_demo.py &

# Start FastAPI on internal port 8000 (app.py contains the FastAPI app object)
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Wait for both to be ready
sleep 3

# nginx proxies the single exposed port 7860:
#   /          → Gradio UI
#   /v1/       → FastAPI
#   /health    → FastAPI
#   /docs      → FastAPI
nginx -g "daemon off;"