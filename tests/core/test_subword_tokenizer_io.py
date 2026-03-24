# tests/core/test_subword_tokenizer_io.py
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig


def test_bpe_save_load_roundtrip(tmp_path):
    texts = ["hello world", "hello there"]
    cfg = SubwordTokenizerConfig(vocab_size=60, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, cfg)

    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    tok.save(vocab_path, merges_path)

    tok2 = SubwordTokenizer.load_from_files(vocab_path=vocab_path, merges_path=merges_path)

    s = "hello world"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.decode(tok.encode(s)) == s


def test_sentencepiece_save_load_roundtrip_backend_artifact(tmp_path):
    texts = ["hello world", "hello there", "world hello"]
    cfg = SubwordTokenizerConfig(vocab_size=80, model_type="sentencepiece")
    tok = SubwordTokenizer.train_from_iterator(texts, cfg)

    art = tmp_path / "tok_sp"
    tok.save(artifact_dir=art)
    tok2 = SubwordTokenizer.load(art)

    s = "hello <|user|> world"
    assert tok2.backend_family == "sentencepiece"
    assert tok2.encode(s) == tok.encode(s)
    assert tok2.decode(tok.encode(s)) == tok.decode(tok.encode(s))
