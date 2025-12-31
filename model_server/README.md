This small FastAPI wrapper exposes the project's text generator via a lightweight OpenAI-compatible HTTP API so OpenWebUI (Open UI) can use it as a remote model.

Endpoints:
- POST /v1/chat/completions - accepts `messages` (list of {role, content}) or `prompt` and returns a chat-style response.
- POST /v1/completions - accepts `prompt` and returns a completion.
- GET /health - health check.

Quick local run (without Docker):

1. Create a virtualenv and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the server:

```bash
uvicorn model_server.main:app --host 0.0.0.0 --port 8000
```

Using with OpenWebUI / Open UI:
- In OpenWebUI settings, set the model API endpoint to `http://<host>:8000` (if OpenWebUI runs in Docker on same machine, use host.docker.internal or the network name). The server implements `/v1/chat/completions` and `/v1/completions`.

Notes:
- The default generator is the `huggingface` generator which may load large models and require GPU/CPU resources. Set `GENERATOR_TYPE` env var to `gemini` to use Google Gemini (requires credentials and `google-genai`).
- The Dockerfile builds an image with all requirements; if your model needs GPU support or special libs (bitsandbytes, CUDA), adapt the Dockerfile accordingly.

