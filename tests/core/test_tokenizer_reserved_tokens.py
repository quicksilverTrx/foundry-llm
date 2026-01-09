import pytest
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig, RESERVED_SPECIAL_TOKENS, _pretokenize

def test_reserved_token_ids_fixed():
    texts = ["Hello <|user|> world", "Test <|assistant|> reply"]
    tok = SubwordTokenizer.train_from_iterator(texts, SubwordTokenizerConfig(vocab_size=200))

    assert tok.token_to_id("<|pad|>") == 0
    assert tok.token_to_id("<|user|>") == 1
    assert tok.token_to_id("<|assistant|>") == 2
    assert tok.token_to_id("<|endoftext|>") == 3

def test_reserved_ids_not_used_by_normal_tokens():
    tok = SubwordTokenizer.train_from_iterator(["hello world"], SubwordTokenizerConfig(vocab_size=100))
    for sym, idx in tok.stoi.items():
        if sym not in RESERVED_SPECIAL_TOKENS:
            assert idx >= len(RESERVED_SPECIAL_TOKENS)

def test_pretokenizer_preserves_special_tokens_atomically():
    toks = _pretokenize("Hello<|user|>world")
    assert "<|user|>" in toks
    assert any(t == "<|user|>" for t in toks)

def test_roundtrip_contains_special_token_exactly():
    texts = ["Hello <|user|> world"]
    tok = SubwordTokenizer.train_from_iterator(texts, SubwordTokenizerConfig(vocab_size=200))
    s = "Hello <|user|> world"
    out = tok.decode(tok.encode(s))
    assert "<|user|>" in out

def test_no_special_syntax_fragments_in_vocab():
    from llm_lab.core.tokenization.subword_tokenizer import (
        SubwordTokenizer, SubwordTokenizerConfig, RESERVED_SPECIAL_TOKENS
    )

    tok = SubwordTokenizer.train_from_iterator(
        ["hello <|user|> world"], SubwordTokenizerConfig(vocab_size=200)
    )

    for sym in tok.stoi.keys():
        # Only special tokens should contain "<|" or "|>"
        if "<|" in sym or "|>" in sym or "|" in sym:
            assert sym in RESERVED_SPECIAL_TOKENS

def test_merges_do_not_include_special_tokens():
    from llm_lab.core.tokenization.subword_tokenizer import (
        SubwordTokenizer, SubwordTokenizerConfig, RESERVED_SPECIAL_TOKENS
    )

    tok = SubwordTokenizer.train_from_iterator(
        ["hello <|user|> world"], SubwordTokenizerConfig(vocab_size=200)
    )

    for a, b in tok.merges:
        assert a not in RESERVED_SPECIAL_TOKENS
        assert b not in RESERVED_SPECIAL_TOKENS