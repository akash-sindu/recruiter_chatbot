import os
import json
from groq import Groq
from pydantic import BaseModel, Field
import sys
from typing import Dict, Any, Literal
import hashlib
from fractions import Fraction
from cachetools import TTLCache

# GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_MODEL = "openai/gpt-oss-120b"

_RESUME_FIT_CACHE: Dict[str, FitGapAnalysis] = {}
_PARSED_JD_CACHE: Dict[str, ParsedJobDescription] = {}
_SESSIONS_TTL_CACHE = TTLCache(maxsize=1000, ttl=7200)


class ParsedJobDescription(BaseModel):
    title: str | None = Field(default=None, description="Job title or role name")
    company: str | None = Field(default=None, description="Company or team name")
    minimum_years_experience: float | None = Field(
        default=None, description="Minimum years of experience required"
    )
    required_skills: list[str] = Field(
        default_factory=list, description="Must-have technical or domain skills"
    )
    preferred_skills: list[str] = Field(
        default_factory=list, description="Nice-to-have or bonus skills"
    )
    key_responsibilities: list[str] = Field(
        default_factory=list, description="Core duties and daily responsibilities"
    )


parsed_jd_schema = ParsedJobDescription.model_json_schema()


class FitGapAnalysis(BaseModel):
    match_score: int | None = Field(
        description="Fit score from 1 to 100 among criteria with profile evidence, or null if none"
    )
    evidence_coverage: int = Field(
        default=0,
        description="Weighted percentage of job criteria with relevant profile evidence",
    )
    strong_matches: list[str] = Field(
        description="Skills or experiences directly matching the JD"
    )
    transferable_skills: list[str] = Field(
        description="Related experience that bridges unlisted requirements"
    )
    potential_gaps: list[str] = Field(
        description="Partially evidenced criteria that need further assessment"
    )
    not_documented: list[str] = Field(
        default_factory=list,
        description="Criteria with no relevant evidence in the candidate profile",
    )
    verdict_summary: str = Field(
        description="1-2 sentence executive verdict for recruiters"
    )


class FitCriterionAssessment(BaseModel):
    criterion: str = Field(description="Exact criterion from the job description")
    category: Literal["required_skill", "responsibility", "preferred_skill"]
    status: Literal["met", "partial", "not_documented"]
    evidence: str = Field(
        description="Supporting candidate-profile evidence, or explain that none is stated"
    )


class FitGapEvaluation(BaseModel):
    criterion_assessments: list[FitCriterionAssessment]
    transferable_skills: list[str] = Field(default_factory=list)
    verdict_summary: str = (
        "Assessment based on the candidate-profile evidence provided."
    )


gap_analysis_schema = FitGapEvaluation.model_json_schema()


def get_session_store(session_id: str) -> Dict[str, Any]:
    """Retrieves or initializes an isolated cache dict for a valid session ID."""
    if not isinstance(session_id, str):
        session_id = str(session_id)

    session_key = session_id.strip()
    if not session_key:
        raise ValueError("session_id must be a non-empty string.")

    store = _SESSIONS_TTL_CACHE.get(session_key)
    if store is None or not isinstance(store, dict):
        store = {}
        _SESSIONS_TTL_CACHE[session_key] = store

    return store


def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is missing.")
    return Groq(api_key=api_key)


def load_candidate_profile(filepath="data/candidate_profile.md") -> str:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Candidate profile file not found."


def parse_and_validate_jd(raw_jd_text: str) -> ParsedJobDescription:
    normalized_jd = " ".join(raw_jd_text.split())
    cache_key = hashlib.sha256(normalized_jd.encode("utf-8")).hexdigest()
    if cache_key in _PARSED_JD_CACHE:
        return _PARSED_JD_CACHE[cache_key]

    client = get_groq_client()
    schema = parsed_jd_schema

    system_prompt = f"""
    You are an expert HR Data Extractor.
    Your task is to parse unstructured job description text into a clean JSON object.
    
    JSON SCHEMA:
    {json.dumps(schema, indent=2)}

    IMPORTANT:
    - Extract explicitly stated information. Use null or empty lists if missing.
    - Do NOT invent qualifications or experience metrics.
    - Return ONLY a valid JSON object matching the schema.
    """

    user_prompt = f"""
    Parse this job description:\n\n{raw_jd_text} """

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content
    if not raw_json:
        raise ValueError("No JSON response received from Groq.")
    data = json.loads(raw_json)
    parsed_jd = ParsedJobDescription(**data)
    _PARSED_JD_CACHE[cache_key] = parsed_jd
    return parsed_jd


def perform_fit_gap_analysis(
    candidate_profile: str, parsed_jd: ParsedJobDescription
) -> FitGapAnalysis:
    client = get_groq_client()
    schema = gap_analysis_schema

    system_prompt = f"""
        You are a fair, evidence-focused recruiter. Assess the candidate against each criterion in the job description.
        Return one assessment for every required skill, preferred skill, and key responsibility, using its exact text.
        Review the entire candidate profile, including the summary, skills, work history, and projects. Judge semantic evidence,
        not keyword overlap: do not require the profile to use the JD's exact wording when it clearly demonstrates the same
        capability or responsibility.

        Apply these status definitions consistently:
        - met: The profile provides clear evidence of the same capability, responsibility, or outcome, including equivalent
            experience described with different wording. A JD-specific named tool, language, or certification is met only if
            that specific item is stated in the profile.
        - partial: The profile shows relevant adjacent or transferable experience, but does not demonstrate the full
            capability or a JD-specific item. Name the demonstrated experience without claiming the missing item.
        - not_documented: The entire profile contains no relevant supporting evidence for the criterion. This does not mean
            that the candidate lacks the capability; it means the profile does not establish it.

            If a criterion combines multiple capabilities, assess its meaningful parts: mark it met only when the core parts
            are supported, partial when relevant parts are supported but others are not, and not_documented only when none are.
            For every met or partial result, include a short exact quote or a precise profile section/fact as evidence. For
            not_documented results, state that the profile does not document relevant evidence.
            Judge only job-related qualifications. Ignore demographic or personal characteristics and proxies such as name,
            age, gender, nationality, disability, family status, or address. Do not mark a criterion missing merely because
            its wording differs. Do not inflate a match based on general knowledge, seniority, or assumptions. Do not calculate
            or guess a numeric score.
    Output a structured assessment strictly following this JSON schema:
    
    JSON SCHEMA:
    {json.dumps(schema, indent=2)}

    CANDIDATE PROFILE:
    {candidate_profile}

    STRUCTURED JOB DESCRIPTION:
    {parsed_jd.model_dump_json(indent=2)}
    
    """
    user_prompt = "Assess every listed job criterion against the candidate profile and return the structured evidence."

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )

    raw_json = response.choices[0].message.content
    if not raw_json:
        raise ValueError("No JSON response received from Groq.")
    data = json.loads(raw_json)
    evaluation = FitGapEvaluation(**data)
    assessments = {
        (
            assessment.category,
            assessment.criterion.strip().casefold(),
        ): assessment.status
        for assessment in evaluation.criterion_assessments
    }

    criteria_by_category = {
        "required_skill": parsed_jd.required_skills,
        "responsibility": parsed_jd.key_responsibilities,
        "preferred_skill": parsed_jd.preferred_skills,
    }
    category_weights = {
        "required_skill": 50,
        "responsibility": 30,
        "preferred_skill": 20,
    }
    status_values = {"met": 2, "partial": 1}
    active_weight = 0
    documented_weight = Fraction(0)
    weighted_score = Fraction(0)
    strong_matches = []
    potential_gaps = []
    not_documented = []

    for category, criteria in criteria_by_category.items():
        if not criteria:
            continue

        category_score = 0
        category_documented_count = 0
        for criterion in criteria:
            status = assessments.get(
                (category, criterion.strip().casefold()), "not_documented"
            )
            if status == "not_documented":
                not_documented.append(criterion)
                continue

            category_documented_count += 1
            category_score += status_values[status]
            if status == "met":
                strong_matches.append(criterion)
            elif status == "partial":
                potential_gaps.append(criterion)
            else:
                potential_gaps.append(criterion)

        category_weight = category_weights[category]
        weighted_score += Fraction(category_weight * category_score, 2 * len(criteria))
        documented_weight += Fraction(
            category_weight * category_documented_count, len(criteria)
        )
        active_weight += category_weight

    raw_score = weighted_score * 100 / documented_weight if documented_weight else None
    match_score = int(raw_score + Fraction(1, 2)) if raw_score is not None else None
    raw_coverage = (
        documented_weight * 100 / active_weight if active_weight else Fraction(0)
    )
    evidence_coverage = int(raw_coverage + Fraction(1, 2))

    return FitGapAnalysis(
        match_score=match_score,
        evidence_coverage=evidence_coverage,
        strong_matches=strong_matches,
        transferable_skills=evaluation.transferable_skills,
        potential_gaps=potential_gaps,
        not_documented=not_documented,
        verdict_summary=evaluation.verdict_summary,
    )


def build_system_prompt(
    candidate_profile: str,
    parsed_jd: ParsedJobDescription | None,
) -> str:
    jd_context = (
        parsed_jd.model_dump_json(indent=2)
        if parsed_jd
        else "No explicit job description file uploaded."
    )
    return f"""
You are an executive candidate assistant speaking directly to recruiters on behalf of the candidate.

Your job is to answer only recruiter-facing questions about the candidate’s experience, skills, projects, and fit for the target role.

IMPORTANT RULES:
1. Use ONLY the candidate profile and the active job description as evidence.
2. Treat the candidate profile as the single source of truth. Conversation history and prior assistant messages are not evidence.
3. Do not combine technologies, responsibilities, or achievements from separate projects or roles into one project or role.
4. The technical skills matrix supports general skill claims; it does not establish that a skill was used in a particular project.
5. For project-specific questions, use only details associated with that project in the profile. Do not assign a tool or API to a project based on its appearance elsewhere in the profile.
6. Do not invent projects, companies, technologies, responsibilities, outcomes, achievements, or metrics.
7. If a detail is not explicitly present in the profile, say:
   "Not explicitly mentioned in the candidate profile."
8. Do not infer missing facts from general software engineering knowledge.
9. If the question is unrelated to the candidate’s experience, role fit, or the job description, politely refuse and redirect to job-related questions.
10. If the profile does not mention a project, skill, tool, metric, or responsibility, do not guess.
11. Distinguish between:
   - explicitly stated facts
   - inferred but supported by the profile
   - not mentioned / unavailable
12. Answer in a confident, recruiter-friendly tone, but remain precise and evidence-based.
13. When asked about fit, compare the profile to the JD only using details actually present in the profile.

CANDIDATE PROFILE:
{candidate_profile}

ACTIVE JOB DESCRIPTION:
{jd_context}

ANSWER FORMAT:
- Be concise and factual
- Use bullet points when helpful
- If a claim is unsupported, say: "Not explicitly mentioned in the candidate profile."
- Never fabricate experience or projects
"""


def _get_resume_hash(candidate_profile: str, parsed_jd: ParsedJobDescription) -> str:
    """Generates a SHA-256 hash for the given resume and job description pair."""
    jd_payload = parsed_jd.model_dump(mode="json")
    jd_string = json.dumps(jd_payload, sort_keys=True, separators=(",", ":"))

    # Hash candidate profile and JD payload together
    composite_payload = f"RESUME:{candidate_profile.strip()}||JD:{jd_string}"
    return hashlib.sha256(composite_payload.encode("utf-8")).hexdigest()


def get_or_create_fit_analysis(
    candidate_profile: str, parsed_jd: ParsedJobDescription, force_refresh: bool = False
) -> FitGapAnalysis:
    """
    Returns a cached fit-gap analysis if the resume was analyzed before in this session.
    Otherwise, invokes Groq, caches the result, and returns the model instance.
    """
    cache_key = _get_resume_hash(candidate_profile, parsed_jd)

    # 1. Check cache for this exact Resume + JD combination
    if not force_refresh and cache_key in _RESUME_FIT_CACHE:
        print(
            f"⚡ [CACHE HIT] Reusing cached fit analysis for Key: {cache_key[:10]}..."
        )
        return _RESUME_FIT_CACHE[cache_key]

    # 2. Cache Miss: Execute LLM Analysis
    print(f"🔍 [CACHE MISS] Running new fit analysis for Key: {cache_key[:10]}...")
    analysis_result = perform_fit_gap_analysis(candidate_profile, parsed_jd)

    # 3. Store in cache associated with this resume
    _RESUME_FIT_CACHE[cache_key] = analysis_result

    return analysis_result


def generate_groq_stream(messages: list):
    client = get_groq_client()

    stream = client.chat.completions.create(
        model=GROQ_MODEL, messages=messages, temperature=0.2, stream=True
    )

    print("\n" + "=" * 50, flush=True)
    print("⚡ [GROQ STREAM STARTED]", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.write("💬 Assistant: ")
    sys.stdout.flush()

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            # 1. Print token directly to backend terminal
            sys.stdout.write(delta)
            sys.stdout.flush()

            # 2. Yield token as Server-Sent Event (SSE) to Postman/Frontend
            yield f"data: {json.dumps({'content': delta})}\n\n"

    print("\n" + "=" * 50, flush=True)
    print("✅ [GROQ STREAM COMPLETED]", flush=True)
    print("=" * 50 + "\n", flush=True)
    yield "data: [DONE]\n\n"
