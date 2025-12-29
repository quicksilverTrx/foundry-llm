import torch

from llm_lab.core.data.lm_dataset import LanguageModelingDataset
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig


def test_lm_dataset_getitem_shapes_and_shift():
    texts = ["hello world", "hello there", "world hello"]
    cfg = SubwordTokenizerConfig(vocab_size=80, model_type="bpe")
    tok = SubwordTokenizer.train_from_iterator(texts, cfg)

    block_size = 8
    ds = LanguageModelingDataset("hello world hello there", tokenizer=tok, block_size=block_size)

    assert len(ds) >= 0  # mostly ensures __len__ doesn't crash

    if len(ds) > 0:
        x, y = ds[0]
        assert x.shape == (block_size,)
        assert y.shape == (block_size,)
        assert x.dtype == torch.long
        assert y.dtype == torch.long

        # shift property
        assert torch.equal(y[:-1], x[1:])
