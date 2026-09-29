from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from services.grok_service import (
    FitGapAnalysis,
    ParsedJobDescription,
    _get_resume_hash,
    build_system_prompt,
    get_session_store,
    get_or_create_fit_analysis,
    parse_and_validate_jd,
    perform_fit_gap_analysis,
)


class GetOrCreateFitAnalysisTests(unittest.TestCase):
    def test_annotation_and_cache_contract(self):
        jd = ParsedJobDescription(
            title="Senior Python Engineer",
            company="Acme",
            minimum_years_experience=3,
            required_skills=["Python", "FastAPI"],
            preferred_skills=["SQL"],
            key_responsibilities=["Build APIs"],
        )

        expected = FitGapAnalysis(
            match_score=88,
            evidence_coverage=100,
            strong_matches=["Python"],
            transferable_skills=["API design"],
            potential_gaps=["System design"],
            not_documented=[],
            verdict_summary="Strong overall fit for the role.",
        )

        hashed = _get_resume_hash("resume text", jd)
        self.assertIsInstance(hashed, str)
        self.assertTrue(hashed)

        with patch(
            "services.grok_service.perform_fit_gap_analysis", return_value=expected
        ):
            result = get_or_create_fit_analysis("resume text", jd)

        self.assertIsInstance(result, FitGapAnalysis)
        self.assertEqual(
            get_or_create_fit_analysis.__annotations__["return"], FitGapAnalysis
        )
        self.assertEqual(result.match_score, 88)

    def test_session_store_requires_valid_session_id(self):
        store = get_session_store(" session-123 ")
        self.assertIsInstance(store, dict)
        store["seen"] = True
        self.assertTrue(get_session_store("session-123")["seen"])

        with self.assertRaises(ValueError):
            get_session_store("   ")

    def test_system_prompt_enforces_grounding(self):
        profile = "Built a Python REST API using Flask and AWS Lambda."
        jd = ParsedJobDescription(
            title="Senior Backend Engineer",
            required_skills=["Python", "FastAPI"],
        )

        prompt = build_system_prompt(profile, jd)

        self.assertIn("single source of truth", prompt.lower())
        self.assertIn(
            "not explicitly mentioned in the candidate profile.", prompt.lower()
        )
        self.assertIn("ACTIVE JOB DESCRIPTION:", prompt)
        self.assertIn(profile, prompt)
        self.assertNotIn("PRE-CALCULATED FIT ANALYSIS:", prompt)
        self.assertIn("Conversation history and prior assistant messages are not evidence", prompt)
        self.assertIn("Do not combine technologies, responsibilities, or achievements from separate projects", prompt)
        self.assertIn("does not establish that a skill was used in a particular project", prompt)

    def test_fit_score_uses_fixed_category_weights(self):
        jd = ParsedJobDescription(
            required_skills=["Python", "FastAPI"],
            key_responsibilities=["Build APIs"],
            preferred_skills=["SQL"],
        )

        response = {
            "criterion_assessments": [
                {
                    "criterion": "Python",
                    "category": "required_skill",
                    "status": "met",
                    "evidence": "Used Python.",
                },
                {
                    "criterion": "FastAPI",
                    "category": "required_skill",
                    "status": "partial",
                    "evidence": "Built Python APIs; framework not stated.",
                },
                {
                    "criterion": "Build APIs",
                    "category": "responsibility",
                    "status": "not_documented",
                    "evidence": "Not stated.",
                },
                {
                    "criterion": "SQL",
                    "category": "preferred_skill",
                    "status": "met",
                    "evidence": "Used SQL.",
                },
            ],
            "transferable_skills": [],
            "verdict_summary": "Partial fit.",
        }

        mock_response = Mock(choices=[Mock(message=Mock(content=json.dumps(response)))])
        with patch("services.grok_service.get_groq_client") as get_client:
            get_client.return_value.chat.completions.create.return_value = mock_response
            result = perform_fit_gap_analysis("Used Python and SQL.", jd)

        self.assertEqual(result.match_score, 82)
        self.assertEqual(result.evidence_coverage, 70)
        self.assertEqual(result.strong_matches, ["Python", "SQL"])
        self.assertEqual(result.potential_gaps, ["FastAPI"])
        self.assertEqual(result.not_documented, ["Build APIs"])

    def test_fit_score_is_unavailable_without_documented_evidence(self):
        jd = ParsedJobDescription(required_skills=["Kubernetes"])
        response = {
            "criterion_assessments": [
                {
                    "criterion": "Kubernetes",
                    "category": "required_skill",
                    "status": "not_documented",
                    "evidence": "Not documented in the profile.",
                }
            ],
        }
        mock_response = Mock(
            choices=[Mock(message=Mock(content=json.dumps(response)))]
        )

        with patch("services.grok_service.get_groq_client") as get_client:
            get_client.return_value.chat.completions.create.return_value = mock_response
            result = perform_fit_gap_analysis("Software engineer profile.", jd)

        self.assertIsNone(result.match_score)
        self.assertEqual(result.evidence_coverage, 0)
        self.assertEqual(result.not_documented, ["Kubernetes"])
        self.assertEqual(
            result.transferable_skills,
            [],
        )
        self.assertEqual(
            result.verdict_summary,
            "Assessment based on the candidate-profile evidence provided.",
        )

    def test_fit_prompt_accounts_for_equivalent_evidence_without_inventing_tools(self):
        jd = ParsedJobDescription(required_skills=["FastAPI"])
        response = Mock(
            choices=[
                Mock(
                    message=Mock(
                        content=json.dumps(
                            {
                                "criterion_assessments": [],
                                "transferable_skills": [],
                                "verdict_summary": "Assessment complete.",
                            }
                        )
                    )
                )
            ]
        )

        with patch("services.grok_service.get_groq_client") as get_client:
            get_client.return_value.chat.completions.create.return_value = response
            perform_fit_gap_analysis("Built REST APIs using Flask.", jd)

        prompt = get_client.return_value.chat.completions.create.call_args.kwargs[
            "messages"
        ][0]["content"]
        normalized_prompt = " ".join(prompt.split())
        self.assertIn("not keyword overlap", normalized_prompt)
        self.assertIn(
            "Do not mark a criterion missing merely because its wording differs",
            normalized_prompt,
        )
        self.assertIn(
            "A JD-specific named tool, language, or certification is met only if",
            normalized_prompt,
        )
        self.assertIn(
            "not_documented: The entire profile contains no relevant supporting evidence",
            normalized_prompt,
        )
        self.assertIn("If a criterion combines multiple capabilities", normalized_prompt)
        self.assertIn("short exact quote", normalized_prompt)
        self.assertIn("Ignore demographic or personal characteristics", normalized_prompt)
        self.assertIn("partial", normalized_prompt)

    def test_repeated_normalized_jd_reuses_parsed_result(self):
        response_content = json.dumps(
            {"title": "Cache Test Role", "required_skills": ["Python"]}
        )
        client = Mock()
        client.chat.completions.create.return_value = Mock(
            choices=[Mock(message=Mock(content=response_content))]
        )

        with patch("services.grok_service.get_groq_client", return_value=client):
            first = parse_and_validate_jd("Unique cache test role requiring Python")
            second = parse_and_validate_jd(
                "  Unique cache test role\nrequiring Python  "
            )

        self.assertEqual(first, second)
        client.chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
