import torch

from llm_lab.core.tokenization import CharTokenizer
from llm_lab.core.data import CharDataset


def test_char_dataset_basic():
    text = "hello"
    tok = CharTokenizer.from_text(text)
    ds = CharDataset(text=text, tokenizer=tok, block_size=3)

    assert len(ds) == 2 

    x0,y0 = ds[0]
    assert x0[1]==y0[0]
    assert x0.shape == torch.Size([3])
    assert y0.shape == torch.Size([3])

    s_x0 = tok.decode(x0.tolist())
    s_y0 = tok.decode(y0.tolist())

    assert s_x0 == "hel"
    assert s_y0 == "ell"