# tests/core/test_tokenizer_save_load_stability.py
import json
from pathlib import Path
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.tokenization.sp16k_tokenizer_artifact import compute_tokenizer_hash

def test_tokenizer_repeated_save_load_is_identical(tmp_path: Path):
    text = "to be or not to be that is the question"
    tok1 = SubwordTokenizer.train_from_iterator([text], SubwordTokenizerConfig(vocab_size=100, model_type="bpe"))

    v1 = tmp_path / "vocab1.txt"
    m1 = tmp_path / "merges1.txt"
    tok1.save(v1, m1)

    tok2 = SubwordTokenizer.load_from_files(v1, m1)

    v2 = tmp_path / "vocab2.txt"
    m2 = tmp_path / "merges2.txt"
    tok2.save(v2, m2)

    tok3 = SubwordTokenizer.load_from_files(v2, m2)

    assert tok1.merges == tok2.merges == tok3.merges
    assert tok1.stoi == tok2.stoi == tok3.stoi
    assert tok1.itos == tok2.itos == tok3.itos


def test_sentencepiece_hash_stable_across_save_load(tmp_path: Path):
    tok1 = SubwordTokenizer.train_from_iterator(
        ["alpha beta gamma", "beta gamma alpha"],
        SubwordTokenizerConfig(vocab_size=96, model_type="sentencepiece"),
    )
    h1 = compute_tokenizer_hash(tok1)

    art = tmp_path / "sp_tok"
    tok1.save(artifact_dir=art)
    tok2 = SubwordTokenizer.load(art)
    h2 = compute_tokenizer_hash(tok2)

    assert h1 == h2


def test_sentencepiece_hash_changes_when_external_map_changes(tmp_path: Path):
    tok = SubwordTokenizer.train_from_iterator(
        ["alpha beta gamma", "beta gamma alpha", "gamma alpha beta"],
        SubwordTokenizerConfig(vocab_size=96, model_type="sentencepiece"),
    )
    art = tmp_path / "sp_tok"
    tok.save(artifact_dir=art)
    h1 = compute_tokenizer_hash(SubwordTokenizer.load(art))

    map_path = art / "external_id_map.json"
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    ext_map = payload["external_to_internal"]
    sorted_keys = sorted(ext_map.keys(), key=int)
    ext_map[sorted_keys[0]], ext_map[sorted_keys[1]] = ext_map[sorted_keys[1]], ext_map[sorted_keys[0]]
    payload["external_to_internal"] = ext_map
    map_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")

    h2 = compute_tokenizer_hash(SubwordTokenizer.load(art))
    assert h1 != h2


def test_sentencepiece_hash_ignores_non_behavioral_meta_fields(tmp_path: Path):
    tok = SubwordTokenizer.train_from_iterator(
        ["alpha beta gamma", "beta gamma alpha"],
        SubwordTokenizerConfig(vocab_size=96, model_type="sentencepiece"),
    )
    art = tmp_path / "sp_tok"
    tok.save(artifact_dir=art)
    h1 = compute_tokenizer_hash(SubwordTokenizer.load(art))

    meta_path = art / "sentencepiece_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["debug_note"] = "non_behavioral"
    meta["tmp_dir"] = "/tmp/example"
    meta_path.write_text(json.dumps(meta, sort_keys=True, indent=2), encoding="utf-8")

    h2 = compute_tokenizer_hash(SubwordTokenizer.load(art))
    assert h1 == h2
