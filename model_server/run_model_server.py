#!/usr/bin/env python3
"""Run the model server from an IDE or CLI using uvicorn.

Usage examples:

# Run on localhost:8000
python run_model_server.py

# Run with reload enabled (useful during development)
python run_model_server.py --reload

# Change host/port
python run_model_server.py --host 127.0.0.1 --port 9000
"""

from argparse import ArgumentParser
import os
import sys


def main():
    parser = ArgumentParser(description="Run the local OpenAI-compatible model server (FastAPI + uvicorn)")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"), help="Host to bind to")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)), help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info"), help="Log level for uvicorn")
    args = parser.parse_args()

    # Ensure project root is on PYTHONPATH so imports work when run from IDE
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import uvicorn
    except Exception as e:
        print("uvicorn is required to run this script. Install requirements with: pip install -r requirements.txt", file=sys.stderr)
        raise

    # Use module path string so uvicorn can reload on code changes
    uvicorn.run("model_server.main:app", host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)


if __name__ == "__main__":
    main()

