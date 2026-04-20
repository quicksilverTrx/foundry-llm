# tests/serving/test_safety_filters.py
from __future__ import annotations

from llm_lab.serving.safety import (
    apply_safety_policy,
    build_refusal_text,
    detect_pii_like,
    detect_profanity_like,
    postprocess_generated_text,
    should_refuse_prompt,
)


def test_detect_pii_like_finds_email_phone_ssn_patterns() -> None:
    text = "Email me at user@example.com or call 415-555-0123 and SSN 123-45-6789"
    flags = detect_pii_like(text)
    assert "pii_email" in flags
    assert "pii_phone" in flags
    assert "pii_ssn" in flags


def test_detect_pii_like_has_reasonable_false_positive_control() -> None:
    text = "The model scored 12345 points and reached step 67890 in training."
    flags = detect_pii_like(text)
    assert flags == []


def test_should_refuse_prompt_returns_reason_codes_for_sensitive_prompt() -> None:
    refuse, codes = should_refuse_prompt("contact: john@example.com")
    assert refuse is True
    assert codes


def test_build_refusal_text_is_stable_and_nonempty() -> None:
    text = build_refusal_text(["pii_email", "pii_phone"])
    assert text.strip() != ""
    assert "john@example.com" not in text


def test_postprocess_generated_text_flags_or_refuses_profanity_like_output() -> None:
    out_text, flags, refused = postprocess_generated_text("this is shit")
    assert refused is True
    assert "profanity_like" in flags
    assert out_text != "this is shit"


def test_postprocess_generated_text_flags_generated_pii_like_output() -> None:
    out_text, flags, refused = postprocess_generated_text("my ssn is 123-45-6789")
    assert refused is True
    assert "pii_ssn" in flags
    assert "123-45-6789" not in out_text


def test_safety_pipeline_does_not_log_raw_sensitive_text() -> None:
    out = apply_safety_policy("email me at jane@example.com", generated_text=None)
    privacy = out["privacy_meta"]
    assert "prompt_hash" in privacy
    assert "prompt_len" in privacy
    assert "output_len" in privacy
    assert "jane@example.com" not in str(privacy)


def test_policy_markers_are_treated_as_normal_text() -> None:
    refuse, codes = should_refuse_prompt("[SAFETY_POLICY_V2] hello")
    assert refuse is False
    assert codes == []
    out_text, flags, refused = postprocess_generated_text("[POSTFILTER_V2] clean text")
    assert refused is False
    assert flags == []
    assert out_text == "[POSTFILTER_V2] clean text"


def test_detect_profanity_like_no_match_on_clean_text() -> None:
    assert detect_profanity_like("have a nice day") == []
