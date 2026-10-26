import json
from unittest.mock import patch

from src.lead_pipeline.llm_responder import LLMResponder


def test_missing_key_returns_deterministic_response_and_logs_reason():
    responder = LLMResponder()

    with patch.dict("os.environ", {}, clear=True):
        with patch("src.lead_pipeline.llm_responder.logging.warning") as warning:
            response = responder.respond({"slots": {"property": "loud"}}, "Fallback.")

    assert response == "Fallback."
    warning.assert_called_once_with("LLM fallback: missing_api_key")


def test_failing_http_call_returns_deterministic_response():
    responder = LLMResponder()

    with patch.dict("os.environ", {"ISL_LLM_API_KEY": "test-key"}):
        with patch(
            "src.lead_pipeline.llm_responder.urlopen",
            side_effect=TimeoutError("request timed out"),
        ) as urlopen:
            with patch("src.lead_pipeline.llm_responder.logging.warning") as warning:
                response = responder.respond(
                    {"slots": {"property": "loud"}},
                    "Fallback.",
                )

    assert response == "Fallback."
    assert urlopen.call_args.kwargs["timeout"] == 5
    assert "test-key" not in str(warning.call_args)


def test_successful_http_response_uses_short_generated_sentence():
    responder = LLMResponder()
    api_response = {
        "choices": [{"message": {"content": "That is loud."}}],
    }
    with patch.dict("os.environ", {"ISL_LLM_API_KEY": "test-key"}):
        with patch("src.lead_pipeline.llm_responder.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(
                api_response
            ).encode("utf-8")
            response = responder.respond(
                {"slots": {"property": "loud"}},
                "Fallback.",
            )

    assert response == "That is loud."


def test_request_limiter_tracks_each_ip_and_resets_after_window():
    from src.lead_pipeline.request_limiter import RequestLimiter

    limiter = RequestLimiter(limit=2, window_seconds=60)

    assert limiter.allow("192.0.2.1", now=100)
    assert limiter.allow("192.0.2.1", now=110)
    assert not limiter.allow("192.0.2.1", now=120)
    assert limiter.allow("192.0.2.2", now=120)
    assert limiter.allow("192.0.2.1", now=160)
