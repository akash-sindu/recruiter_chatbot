import os
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

from services.document_parser import extract_text_from_file
from services.grok_service import (
    load_candidate_profile,
    parse_and_validate_jd,
    get_or_create_fit_analysis,
    build_system_prompt,
    generate_groq_stream,
    get_session_store,
)

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Application State
CANDIDATE_PROFILE = load_candidate_profile()


def get_session_state(session_id: str | None = None) -> dict:
    """Return the per-session state bucket, creating it on demand."""
    effective_session_id = session_id or "default"
    return get_session_store(effective_session_id)


def process_jd_text(raw_text: str, session_id: str | None = None) -> dict:
    """Executes the Pydantic extraction + fit-gap prompt chain for one session."""
    session_store = get_session_state(session_id)

    # Chain Step 1: Parse & Validate raw text into Pydantic model
    active_parsed_jd = parse_and_validate_jd(raw_text)
    session_store["ACTIVE_PARSED_JD"] = active_parsed_jd

    # Chain Step 2: Execute pre-analysis using candidate profile + parsed JD
    active_fit_analysis = get_or_create_fit_analysis(
        CANDIDATE_PROFILE, active_parsed_jd
    )
    session_store["ACTIVE_FIT_ANALYSIS"] = active_fit_analysis

    return {
        "status": "success",
        "parsed_jd": active_parsed_jd.model_dump(),
        "fit_analysis": active_fit_analysis.model_dump(),
    }


@app.route("/api/health", methods=["GET"])
def health():
    session_id = request.args.get("session_id", "default")
    session_store = get_session_state(session_id)
    return jsonify(
        {
            "status": "healthy",
            "profile_loaded": bool(CANDIDATE_PROFILE),
            "jd_parsed": session_store.get("ACTIVE_PARSED_JD") is not None,
            "fit_analysis_ready": session_store.get("ACTIVE_FIT_ANALYSIS") is not None,
        }
    )


@app.route("/api/upload-jd-file", methods=["POST"])
def upload_jd_file():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id", "default")
    try:
        raw_text = extract_text_from_file(file)
        result = process_jd_text(raw_text, session_id=session_id)
        result["filename"] = file.filename
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload-jd-text", methods=["POST"])
def upload_jd_text():

    if request.method == "OPTIONS":
        return "", 200
    data = request.json or {}
    text = data.get("text", "").strip()
    session_id = data.get("session_id", "default")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        result = process_jd_text(text, session_id=session_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    data = request.json or {}
    session_id = data.get("session_id", "default")
    user_message = data.get("message")
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    session_store = get_session_state(session_id)
    active_parsed_jd = session_store.get("ACTIVE_PARSED_JD")

    # Keep model-generated fit summaries out of the chat evidence context.
    system_prompt = build_system_prompt(
        candidate_profile=CANDIDATE_PROFILE,
        parsed_jd=active_parsed_jd,
    )

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.get("role"), "content": turn.get("content")})
    messages.append({"role": "user", "content": user_message})

    try:
        return Response(generate_groq_stream(messages), mimetype="text/event-stream")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 50000))
    app.run(host="0.0.0.0", port=port, debug=True)
