# tests/serving/test_tiktoken_wrapper.py
from __future__ import annotations

import pytest

from llm_lab.core.tokenization.tiktoken_wrapper import TiktokenWrapper
from llm_lab.serving._shared import resolve_eos_token_id


@pytest.fixture()
def wrapper() -> TiktokenWrapper:
    return TiktokenWrapper(encoding_name="gpt2")


def test_encode_decode_roundtrip(wrapper: TiktokenWrapper) -> None:
    text = "hello world"
    assert wrapper.decode(wrapper.encode(text)) == text


def test_eos_token_id(wrapper: TiktokenWrapper) -> None:
    assert wrapper.eos_token_id == 50256


def test_pad_token_id(wrapper: TiktokenWrapper) -> None:
    assert wrapper.pad_token_id == 50256


def test_token_to_id_endoftext(wrapper: TiktokenWrapper) -> None:
    assert wrapper.token_to_id("<|endoftext|>") == 50256


def test_token_to_id_rejects_multi_token_string(wrapper: TiktokenWrapper) -> None:
    with pytest.raises(ValueError, match="does not map to a single token ID"):
        wrapper.token_to_id("hello world")


def test_resolve_eos_works(wrapper: TiktokenWrapper) -> None:
    assert resolve_eos_token_id(wrapper) == 50256


def test_decode_filters_out_of_vocab(wrapper: TiktokenWrapper) -> None:
    valid_id = 284  # " the" in GPT-2
    valid_text = wrapper.decode([valid_id])
    # IDs in model padding range (50257-50303) must be silently dropped
    assert wrapper.decode([50300, valid_id]) == valid_text
    assert wrapper.decode([50257, valid_id, 50303]) == valid_text


def test_decode_empty_list(wrapper: TiktokenWrapper) -> None:
    assert wrapper.decode([]) == ""


def test_decode_eos_token(wrapper: TiktokenWrapper) -> None:
    # EOS (50256) is a valid tiktoken token and should decode
    result = wrapper.decode([50256])
    assert isinstance(result, str)


def test_encode_with_special_tokens(wrapper: TiktokenWrapper) -> None:
    ids = wrapper.encode("<|endoftext|>")
    assert ids == [50256]


def test_vocab_size(wrapper: TiktokenWrapper) -> None:
    assert wrapper.vocab_size == 50257
