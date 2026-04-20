# llm_lab/serving/safety.py
from __future__ import annotations

import re

from llm_lab.serving._shared import sha256_text


_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"\b(?:\+1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

_PROFANITY_TERMS = ("fuck", "shit", "bitch", "asshole")


def detect_pii_like(text: str) -> list[str]:
    flags: list[str] = []
    if _EMAIL_RE.search(text):
        flags.append("pii_email")
    if _PHONE_RE.search(text):
        flags.append("pii_phone")
    if _SSN_RE.search(text):
        flags.append("pii_ssn")
    return flags


def detect_profanity_like(text: str) -> list[str]:
    lowered = text.lower()
    out: list[str] = []
    for term in _PROFANITY_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            out.append("profanity_like")
            break
    return out


def should_refuse_prompt(text: str) -> tuple[bool, list[str]]:
    codes = detect_pii_like(text)
    return (len(codes) > 0, codes)


def build_refusal_text(reason_codes: list[str]) -> str:
    if not reason_codes:
        return "I cannot comply with this request due to safety policy."
    joined = ", ".join(sorted(set(reason_codes)))
    return f"I cannot provide that response due to safety policy ({joined})."


def postprocess_generated_text(text: str) -> tuple[str, list[str], bool]:
    flags = sorted(set(detect_pii_like(text) + detect_profanity_like(text)))
    if flags:
        return build_refusal_text(flags), flags, True
    return text, [], False


def apply_safety_policy(prompt_text: str, generated_text: str | None = None) -> dict:
    prompt_refused, prompt_codes = should_refuse_prompt(prompt_text)
    if prompt_refused:
        out_text = build_refusal_text(prompt_codes)
        flags = prompt_codes
        refusal = True
    elif generated_text is not None:
        out_text, flags, refusal = postprocess_generated_text(generated_text)
    else:
        out_text, flags, refusal = "", [], False
    return {
        "prompt_refused": prompt_refused,
        "safety_flags": flags,
        "refusal_applied": refusal,
        "output_text": out_text,
        "privacy_meta": {"prompt_hash": sha256_text(prompt_text), "prompt_len": len(prompt_text), "output_len": len(out_text)},
    }
