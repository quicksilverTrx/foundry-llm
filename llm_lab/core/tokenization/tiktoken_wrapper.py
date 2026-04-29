# llm_lab/core/tokenization/tiktoken_wrapper.py
"""Thin wrapper making tiktoken.Encoding satisfy the serving tokenizer interface.

Addresses:
- G1: decode() filters IDs >= n_vocab to avoid KeyError on model padding range (50257-50303)
- G2: eos_token_id and token_to_id() exposed so resolve_eos_token_id() returns 50256
- G3: pad_token_id set to EOS (standard convention when no dedicated pad token exists)
"""
from __future__ import annotations

from typing import Sequence

import tiktoken


class TiktokenWrapper:
    def __init__(self, encoding_name: str = "gpt2", eos_token_id: int = 50256):
        self._enc = tiktoken.get_encoding(encoding_name)
        self.eos_token_id = eos_token_id
        self.pad_token_id = eos_token_id
        self.vocab_size = self._enc.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, allowed_special="all")

    def decode(self, ids: Sequence[int]) -> str:
        n = self._enc.n_vocab
        return self._enc.decode([i for i in ids if 0 <= i < n])

    def token_to_id(self, token: str) -> int:
        ids = self._enc.encode(token, allowed_special="all")
        if len(ids) != 1:
            raise ValueError(f"{token!r} does not map to a single token ID")
        return ids[0]
