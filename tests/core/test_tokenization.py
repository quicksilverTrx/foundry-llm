#tests/core/test_tokenization.py
from llm_lab.core.tokenization import CharTokenizer


def test_char_tokenizer_round_trip():
    text = "hello world"
    tok = CharTokenizer.from_text(text)

    for s in ["hello", "world", "helo", ""]:
        ids = tok.encode(s)
        s2 = tok.decode(ids)
        assert s2 == s
def test_char_tokenizer_with_unk():
    text = "hello world"
    tok = CharTokenizer.from_text(text, add_unk=True)

    ids = tok.encode("halo")  
    s2 = tok.decode(ids)

    assert s2 == "h<unk>lo" 

