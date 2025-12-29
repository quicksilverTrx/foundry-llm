import pytest
from llm_lab.core.data.lm_dataset import LanguageModelingDataset
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig

def test_contiguous_token_split_has_no_overlap_positions():
    # synthetic unique token stream: overlap-by-value cannot happen
    ids = list(range(1000))
    split = 800
    train_ids = ids[:split]
    val_ids = ids[split:]

    assert train_ids[-1] == split - 1
    assert val_ids[0] == split
    assert set(train_ids).isdisjoint(set(val_ids))
