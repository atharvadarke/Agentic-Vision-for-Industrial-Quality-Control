"""
tests/test_agent_failsafe.py — Agent LLM Fail-Safe Tests  (HIGH RISK GAP #3)

Tests the three failure modes in src/agent.py run_agent():
  1. JSON parse failure  → conservative fallback dict returned
  2. Missing required keys in LLM output → conservative fallback dict
  3. LLM API/network exception → conservative fallback dict

In all cases the contract is:
  - defect_confirmed = True   (never pass a defective part when the agent fails)
  - severity_score  = min(abs(anomaly_score), 1.0)
  - _error key present in the returned dict

Also tests query_historical_defects() against a real (temp) SQLite DB:
  - Known keyword returns matches
  - Unknown keyword returns the no-match fallback string (not an exception)
  - DB is seeded on first call

These tests run without a real GROQ_API_KEY by patching the LLM call.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _make_llm_response(content: str):
    """Build a mock LLM response object with a .content attribute."""
    mock_resp = MagicMock()
    mock_resp.content = content
    return mock_resp


VALID_LLM_JSON = json.dumps({
    "defect_confirmed": True,
    "severity_score": 0.82,
    "historical_analogy": "[Q3 2024] surface_scratch severity 0.82",
    "recommended_action": "Halt conveyor belt and inspect belt tension.",
})

ANOMALY_SCORE = -0.35


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_agent() — happy path
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAgentHappyPath:

    def test_valid_llm_response_returns_structured_dict(self, tmp_path):
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.return_value = _make_llm_response(VALID_LLM_JSON)

            from src.agent import run_agent
            report = run_agent("/path/to/part.jpg", ANOMALY_SCORE)

        assert report["defect_confirmed"] is True
        assert 0.0 <= report["severity_score"] <= 1.0
        assert "historical_analogy" in report
        assert "recommended_action" in report
        assert "_error" not in report

    def test_llm_called_once_per_invocation(self, tmp_path):
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.return_value = _make_llm_response(VALID_LLM_JSON)

            from src.agent import run_agent
            run_agent("/path/to/part.jpg", ANOMALY_SCORE)

        llm_instance.invoke.assert_called_once()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_agent() — FAIL-SAFE: JSON parse failure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAgentJsonParseFailure:

    def _run_with_bad_content(self, content, tmp_path):
        """Helper: patch LLM to return bad content and call run_agent."""
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.return_value = _make_llm_response(content)

            from src.agent import run_agent
            return run_agent("/path/to/part.jpg", ANOMALY_SCORE)

    def test_plain_text_response_triggers_failsafe(self, tmp_path):
        """
        HIGH RISK: LLM returns prose instead of JSON.
        Must return a conservative dict — never raise JSONDecodeError to the caller.
        """
        report = self._run_with_bad_content(
            "The part shows signs of surface corrosion. Please inspect immediately.",
            tmp_path,
        )
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_markdown_wrapped_json_triggers_failsafe(self, tmp_path):
        """LLM wraps JSON in ```json ... ``` markdown — json.loads fails."""
        content = f"```json\n{VALID_LLM_JSON}\n```"
        report = self._run_with_bad_content(content, tmp_path)
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_empty_response_triggers_failsafe(self, tmp_path):
        report = self._run_with_bad_content("", tmp_path)
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_partial_json_triggers_failsafe(self, tmp_path):
        report = self._run_with_bad_content('{"defect_confirmed": true', tmp_path)  # unclosed
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_failsafe_severity_capped_at_1(self, tmp_path):
        """Severity in fail-safe = min(abs(anomaly_score), 1.0)."""
        report = self._run_with_bad_content("not json", tmp_path)
        assert report["severity_score"] == pytest.approx(
            min(abs(ANOMALY_SCORE), 1.0), abs=1e-5
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_agent() — FAIL-SAFE: missing required keys in parsed JSON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAgentMissingKeys:

    def _run_with_partial_json(self, partial_dict, tmp_path):
        content = json.dumps(partial_dict)
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.return_value = _make_llm_response(content)

            from src.agent import run_agent
            return run_agent("/path/part.jpg", ANOMALY_SCORE)

    def test_missing_defect_confirmed_triggers_failsafe(self, tmp_path):
        partial = {
            "severity_score": 0.8,
            "historical_analogy": "Some analogy text here.",
            "recommended_action": "Halt line.",
        }
        report = self._run_with_partial_json(partial, tmp_path)
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_missing_severity_score_triggers_failsafe(self, tmp_path):
        partial = {
            "defect_confirmed": True,
            "historical_analogy": "Some analogy text.",
            "recommended_action": "Halt line immediately.",
        }
        report = self._run_with_partial_json(partial, tmp_path)
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_missing_recommended_action_triggers_failsafe(self, tmp_path):
        partial = {
            "defect_confirmed": True,
            "severity_score": 0.75,
            "historical_analogy": "Some past defect analogy.",
        }
        report = self._run_with_partial_json(partial, tmp_path)
        assert report["defect_confirmed"] is True
        assert "_error" in report

    def test_all_keys_present_no_failsafe(self, tmp_path):
        """When all 4 required keys are present, no fallback should trigger."""
        full = {
            "defect_confirmed": True,
            "severity_score": 0.75,
            "historical_analogy": "Surface scratch Q3 2024.",
            "recommended_action": "Halt conveyor belt for inspection.",
        }
        report = self._run_with_partial_json(full, tmp_path)
        assert "_error" not in report


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# run_agent() — FAIL-SAFE: API / network exception
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestRunAgentApiException:

    def test_llm_api_exception_triggers_failsafe(self, tmp_path):
        """
        HIGH RISK: If the Groq API call raises (network timeout, auth error, etc.),
        run_agent() must NOT propagate the exception — it must return the conservative
        fail-safe dict so the /inspect-part endpoint still returns a Fail response.
        """
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.side_effect = ConnectionError("Groq API unreachable")

            from src.agent import run_agent
            report = run_agent("/path/to/part.jpg", ANOMALY_SCORE)

        assert report["defect_confirmed"] is True
        assert "_error" in report
        assert "LLM API error" in report["_error"]

    def test_llm_auth_error_triggers_failsafe(self, tmp_path):
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "bad-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.side_effect = PermissionError("401 Unauthorized")

            from src.agent import run_agent
            report = run_agent("/path/to/part.jpg", -0.5)

        assert report["defect_confirmed"] is True
        assert report["severity_score"] == pytest.approx(0.5, abs=1e-5)

    def test_exception_failsafe_never_raises(self, tmp_path):
        """The failsafe must swallow all exceptions — no unhandled raise."""
        with patch("src.agent.ChatGroq") as MockChatGroq, \
             patch("src.agent.settings") as mock_cfg:
            mock_cfg.groq_api_key = "test-key"
            mock_cfg.groq_model = "llama-3.3-70b-versatile"
            mock_cfg.llm_temperature = 0.1
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            llm_instance = MockChatGroq.return_value
            llm_instance.invoke.side_effect = RuntimeError("Unexpected internal error")

            from src.agent import run_agent
            # Must not raise — must return a dict
            result = run_agent("/path/part.jpg", -0.2)
        assert isinstance(result, dict)
        assert result["defect_confirmed"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# query_historical_defects()  —  SQLite DB layer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class TestQueryHistoricalDefects:

    def test_known_keyword_returns_matches(self, tmp_path):
        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            from src.agent import query_historical_defects
            result = query_historical_defects("scratch")
        assert "Historical defect matches found" in result
        assert "surface_scratch" in result

    def test_unknown_keyword_returns_no_match_message(self, tmp_path):
        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            from src.agent import query_historical_defects
            result = query_historical_defects("xyzzy_nonexistent_defect_type_9999")
        assert "No exact historical match" in result
        assert "novel defect" in result

    def test_db_auto_seeded_on_first_call(self, tmp_path):
        """DB must be auto-created and seeded without any prior setup."""
        db_path = str(tmp_path / "defect_history.db")
        assert not os.path.exists(db_path), "DB should not exist before first call"

        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = db_path
            from src.agent import query_historical_defects
            query_historical_defects("scratch")

        assert os.path.exists(db_path), "DB must be created on first call"

    def test_returns_at_most_3_results(self, tmp_path):
        """LIMIT 3 in the query — no more than 3 matches returned."""
        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            from src.agent import query_historical_defects
            # 'a' matches most records — should still return ≤3
            result = query_historical_defects("a")
        lines = [l for l in result.split("\n") if l.strip().startswith("-")]
        assert len(lines) <= 3

    def test_results_ordered_by_severity_descending(self, tmp_path):
        """Most severe historical match must appear first."""
        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            from src.agent import query_historical_defects
            result = query_historical_defects("a")

        if "Historical defect matches found" in result:
            lines = [l for l in result.split("\n") if "severity:" in l]
            if len(lines) >= 2:
                # Extract severity values and check descending order
                import re
                severities = [float(re.search(r"severity: ([\d.]+)", l).group(1))
                              for l in lines if re.search(r"severity: ([\d.]+)", l)]
                assert severities == sorted(severities, reverse=True)

    def test_does_not_raise_on_empty_search_term(self, tmp_path):
        with patch("src.agent.settings") as mock_cfg:
            mock_cfg.db_path = str(tmp_path / "defect_history.db")
            from src.agent import query_historical_defects
            result = query_historical_defects("")
        assert isinstance(result, str)
