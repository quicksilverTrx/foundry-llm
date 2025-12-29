import pytest

from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig


def test_bpe_train_is_deterministic():
    texts = ["hello world", "hello there", "world hello"]
    cfg = SubwordTokenizerConfig(vocab_size=50, model_type="bpe")

    tok1 = SubwordTokenizer.train_from_iterator(texts, cfg)
    tok2 = SubwordTokenizer.train_from_iterator(texts, cfg)

    assert tok1.stoi == tok2.stoi
    assert tok1.itos == tok2.itos
    assert tok1.merges == tok2.merges


def test_bpe_encode_decode_roundtrip_simple_spacing():
    texts = ["hello world", "hello there"]
    cfg = SubwordTokenizerConfig(vocab_size=50, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, cfg)

    s = "hello world"
    ids = tok.encode(s)
    s2 = tok.decode(ids)

    # Your tokenizer splits on whitespace; this should hold for single-space inputs.
    assert s2 == s


def test_bpe_unknown_symbol_raises():
    texts = ["hello"]
    cfg = SubwordTokenizerConfig(vocab_size=30, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, cfg)

    with pytest.raises(KeyError):
        tok.encode("z")  # unseen char -> should fail loudly in your P1 design
