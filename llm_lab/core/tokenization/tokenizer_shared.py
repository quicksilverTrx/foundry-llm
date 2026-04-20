from __future__ import annotations

import re
from typing import Dict, List

RESERVED_SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|user|>": 1,
    "<|assistant|>": 2,
    "<|endoftext|>": 3,
}
ID2SPECIAL = {i: s for s, i in RESERVED_SPECIAL_TOKENS.items()}

SPECIAL_RE = re.compile(r"<\|[^|]+?\|>")
_BASIC_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]+", re.UNICODE)

# SP16K bridge START: sp16k pretokenizer spec contract
PRETOKENIZER_SPEC_VERSION = "tinyllama_p15.v1"


def pretokenize(text: str) -> List[str]:
    out: List[str] = []
    i = 0
    for m in SPECIAL_RE.finditer(text):
        left = text[i : m.start()]
        out.extend(_BASIC_TOKEN_RE.findall(left))
        out.append(m.group(0))
        i = m.end()
    out.extend(_BASIC_TOKEN_RE.findall(text[i:]))
    return [t for t in out if t and not t.isspace()]


def get_pretokenizer_spec() -> Dict[str, str]:
    return {
        "version": PRETOKENIZER_SPEC_VERSION,
        "special_token_pattern": SPECIAL_RE.pattern,
        "basic_token_pattern": _BASIC_TOKEN_RE.pattern,
        "special_token_format": "<|name|>",
        "whitespace_policy": "drop pure-whitespace tokens",
    }


# SP16K bridge END: sp16k pretokenizer spec contract
