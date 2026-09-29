# Recruiter Chatbot

A Flask API that helps recruiters compare a candidate profile with a job description. It uses Groq models to structure job descriptions, assess evidence for role criteria, and stream recruiter-facing answers grounded in the candidate profile.

## Features

- Accept job descriptions as text or as PDF, DOCX, and TXT files.
- Extract role details into structured fields such as required skills, preferred skills, and responsibilities.
- Assess each criterion against the candidate profile, separating strong matches, transferable skills, potential gaps, and items not documented.
- Chat with a streaming response based only on the candidate profile and the active job description.
- Keep job-description state isolated by `session_id` (session state expires after two hours).

The candidate profile is loaded from `data/candidate_profile.md` when the app starts. That file is intentionally excluded by `.gitignore` because it contains personal data. Supply your own authorized profile locally; never commit real resumes or candidate records. Job descriptions, profile content, and recruiter chat messages are sent to Groq for processing, so use synthetic data unless you have approval to share it with that provider.

## Requirements

- Python 3.10 or newer
- A Groq API key

## Setup

From the project directory, create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Set `GROQ_API_KEY` in a local `.env` file in the project root:

```dotenv
GROQ_API_KEY=your_groq_api_key
```

Do not commit `.env` or expose the API key. The service currently uses the `openai/gpt-oss-120b` model through Groq. The API has no authentication and enables permissive CORS; keep it on a trusted development machine and do not expose it to the public internet.

## Run

```bash
python app.py
```

The API listens on `http://localhost:50000` by default. Set `PORT` to use a different port. The built-in Flask server runs in debug mode and is intended for local development, not production deployment.

## API

All endpoints are prefixed with `/api`. JSON requests should use `Content-Type: application/json`. If omitted, `session_id` defaults to `default`.

### Health check

```bash
curl 'http://localhost:50000/api/health?session_id=demo'
```

The response reports service health and whether a job description and fit analysis are available for that session.

### Submit job-description text

```bash
curl -X POST http://localhost:50000/api/upload-jd-text \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","text":"Senior Python Engineer. Required: Python, Flask, and API design."}'
```

The response includes `parsed_jd` and `fit_analysis`. The fit analysis includes `match_score` (scored among criteria with documented evidence) and `evidence_coverage` (the weighted share of criteria with relevant profile evidence). A missing profile detail means it is not documented, not necessarily that the candidate lacks that skill.

### Upload a job-description file

Supported extensions are `.pdf`, `.docx`, and `.txt`.

```bash
curl -X POST http://localhost:50000/api/upload-jd-file \
  -F 'session_id=demo' \
  -F 'file=@job-description.pdf'
```

### Stream a recruiter chat response

Submit the same `session_id` used to upload the job description. The endpoint streams Server-Sent Events; each content event is JSON with a `content` field, followed by `data: [DONE]`.

```bash
curl -N -X POST http://localhost:50000/api/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"Which requirements have the strongest evidence?","history":[]}'
```

`history` is an optional array of `{ "role": "user" | "assistant", "content": "..." }` messages. The model is instructed not to treat chat history as evidence about the candidate.

## Tests

Run the unit tests from the project root:

```bash
python -m unittest discover -s test
```

## Project structure

```text
.
├── app.py
├── data/
│   └── candidate_profile.md
├── services/
│   ├── document_parser.py
│   └── grok_service.py
└── test/
    └── test_grok_service.py
```
