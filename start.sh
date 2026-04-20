#!/bin/bash
set -e
cd /app
exec /app/.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
