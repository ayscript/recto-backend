# Recto AI Backend

This repository contains a small FastAPI backend and an agent module used to provide a chat-based AI interface. It integrates with Supabase for authentication and persistence and relies on environment variables for credentials.

**Status:** prototype / development

**Key features**
- REST API with authentication (Supabase)
- Chat endpoint that forwards messages to an agent (`agent/chatbot.py`)
- Endpoints for signup/login and retrieving user profile and sessions

**Requirements**
- Python 3.14 or newer (see `pyproject.toml`)
- See `pyproject.toml` for declared dependencies (FastAPI, uvicorn, supabase, langchain, etc.)

Project structure (important files)
- `main.py` — FastAPI app and HTTP routes (/health, /chat, /history, /sessions, /signup, /login, /get_profile)
- `auth.py` — HTTP Bearer dependency that validates tokens with Supabase
- `database.py` — Supabase client initialization (reads `SUPABASE_URL` and `SUPABASE_ANON_KEY` from env)
- `agent/` — contains the agent and chatbot logic (`agent/chatbot.py`)
- `pyproject.toml` — project metadata and dependencies

Environment variables
- `SUPABASE_URL` — Supabase project URL
- `SUPABASE_ANON_KEY` — Supabase anon/public key used in `database.py`
- Add any other keys required by `agent/chatbot.py` (LLM API keys, etc.) to a `.env` file at the project root. The code calls `load_dotenv()` so `.env` will be loaded if present.

Quick start (Windows)
1. Create a virtual environment and activate it:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies

Option A — install editable package (recommended when using `pyproject.toml`):

```bash
uv sync
```

Option B — manually install the main dependencies shown in `pyproject.toml`:

```bash
uv add install fastapi uvicorn supabase-python python-dotenv pydantic
```

3. Create a `.env` file with at least the Supabase variables:

```
SUPABASE_URL=<your-supabase-url>
SUPABASE_ANON_KEY=<your-anon-key>
GOOGLE_API_KEY=<your-gemini-api-key>
SUPABASE_DB_URL
```

4. Run the app (development):

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

API usage examples
- Health check:

```bash
curl http://127.0.0.1:8000/health
```

- Chat (example request body):

```bash
curl -X POST http://127.0.0.1:8000/chat \
	-H "Authorization: Bearer <TOKEN>" \
	-H "Content-Type: application/json" \
	-d '{"session_id":"session-123","message":"Hello"}'
```

- Signup / Login endpoints accept JSON bodies per the Pydantic schemas in `main.py`.

Notes & next steps
- Ensure `agent/chatbot.py` has the LLM/API keys it requires configured in the environment.

