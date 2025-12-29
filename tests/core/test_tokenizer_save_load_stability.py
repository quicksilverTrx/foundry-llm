from pathlib import Path
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig

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
