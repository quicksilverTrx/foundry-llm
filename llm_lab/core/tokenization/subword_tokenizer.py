# llm_lab/core/tokenization/subword_tokenizer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Dict, Optional, Literal, Union, Tuple, Iterable
import json
import re

Symbol = str
Pair = Tuple[Symbol, Symbol]
PathLike = Union[str, Path]
RESERVED_SPECIAL_TOKENS: Dict[str, int] = {
    "<|pad|>": 0,
    "<|user|>": 1,
    "<|assistant|>": 2,
    "<|endoftext|>": 3,
}
ID2SPECIAL = {i: s for s, i in RESERVED_SPECIAL_TOKENS.items()}
SPECIAL_RE = re.compile(r"<\|[^|]+?\|>")  
_BASIC_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\w\s]+", re.UNICODE)

def _pretokenize(text:str) -> List[str]:
    out : List[str] = []
    i = 0
    for m in SPECIAL_RE.finditer(text):
        left = text[i:m.start()]
        out.extend(_BASIC_TOKEN_RE.findall(left))
        out.append(m.group(0))
        i = m.end()
    out.extend(_BASIC_TOKEN_RE.findall(text[i:]))
    return [t for t in out if t and not t.isspace()]

@dataclass
class SubwordTokenizerConfig:
    vocab_size: int
    model_type: Literal["bpe", "sentencepiece"] = "bpe"



class SubwordTokenizer:
    """
    Abstraction over a subword tokenizer (BPE or SentencePiece).

    we implement a simple char-level BPE backend:
      - words split on whitespace
      - chars + '</w>' for end-of-word
      - greedy BPE merges
    """


    def __init__(
        self,
        stoi: Dict[Symbol, int],
        itos: Dict[int, Symbol],
        config: SubwordTokenizerConfig,
        merges: Optional[Sequence[Pair]] = None,
    ) -> None:
        if config.model_type != "bpe":
  
            raise NotImplementedError("Only 'bpe' backend is implemented.")

        self.stoi = stoi
        self.itos = itos
        self.config = config

        self.merges: List[Pair] = list(merges) if merges is not None else []
        self.pair2rank: Dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}

    def token_to_id(self, tok: str) -> int:
        return self.stoi[tok]

    def id_to_token(self, idx: int) -> str:
        return self.itos[idx]

    def is_special(self, tok: str) -> bool:
        return tok in RESERVED_SPECIAL_TOKENS
    

    @classmethod
    def train_from_iterator(
        cls,
        texts: Sequence[str],
        config: SubwordTokenizerConfig,
    ) -> "SubwordTokenizer":
        """
        Train a simple char-level BPE tokenizer from an in-memory collection
        of texts.
        """
        # Build initial corpus (list of word-symbol sequences) + vocab
        corpus_words, vocab = cls._build_initial_corpus(texts)

        merges: List[Pair] = []
        #Iteratively pick and merge the most frequent pair
        while len(vocab) < config.vocab_size:
            if len(vocab) %100 ==0:
                print(f"Length of vocabulary is {len(vocab)}")
            pair_counts = cls._count_pairs(corpus_words)
            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_count < 2:  # optional early stop
                break

            merges.append(best_pair)
            new_symbol = best_pair[0] + best_pair[1]
            vocab.add(new_symbol)

            cls._merge_pair_in_corpus(corpus_words, best_pair, new_symbol)

        #Build vocab -> ids mapping in a deterministic order
        reserved_items = sorted(RESERVED_SPECIAL_TOKENS.items(),key = lambda kv : kv[1])
        reserved_syms = [s for s,_ in reserved_items]


        learned_vocab = sorted(sym for sym in vocab if sym not in RESERVED_SPECIAL_TOKENS)
        symbols = reserved_syms + learned_vocab
        for s,idx in RESERVED_SPECIAL_TOKENS.items():
            assert symbols[idx] == s, (s,idx,symbols[idx])


        stoi = {sym: i for i, sym in enumerate(symbols)}
        itos = {i: sym for sym, i in stoi.items()}

        return cls(stoi=stoi, itos=itos, config=config, merges=merges)

    @classmethod
    def load_from_files(
        cls,
        vocab_path: PathLike,
        merges_path: Optional[PathLike] = None,
    ) -> "SubwordTokenizer":
        """
        Load a trained BPE tokenizer from disk.

        Format:
          - vocab file: one token per line (id = line index)
          - merges file: each line 'a b' for a merge (a,b) in order
        """
        if merges_path is None:
            raise ValueError("merges_path is required for BPE load.")

        symbols = cls._load_vocab(vocab_path)
        merges = cls._load_merges(merges_path)

        stoi = {sym: i for i, sym in enumerate(symbols)}
        itos = {i: sym for i, sym in enumerate(symbols)}

        config = SubwordTokenizerConfig(
            vocab_size=len(symbols),
            model_type="bpe",
        )
        return cls(stoi=stoi, itos=itos, config=config, merges=merges)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,  # unused in this minimal P1 version
    ) -> List[int]:
        """
        Convert string -> list of token ids using BPE.
        Simple whitespace tokenization: split on spaces, then apply BPE per word.
        """
        ids: List[int] = []
        for tok in _pretokenize(text):
            if tok in RESERVED_SPECIAL_TOKENS:
                ids.append(self.stoi[tok])
                continue

            for sym in self._bpe_encode_word(tok):
                try:
                    ids.append(self.stoi[sym])
                except KeyError:
                    raise KeyError(f"Unknown BPE symbol {sym!r}") from None
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """
        Convert list of token ids back into a string.
        Reverse the '</w>' convention: symbols ending with '</w>' close a word.
        """
        symbols = (self.itos[i] for i in ids)
        words: List[str] = []
        current = ""

        for sym in symbols:
            if sym in RESERVED_SPECIAL_TOKENS:
                if current:
                    words.append(current)
                    current = ""
                words.append(sym)
                continue

            if sym.endswith("</w>"):
                current += sym[: -len("</w>")]
                words.append(current)
                current = ""
            else:
                current += sym

        if current:
            words.append(current)

        return " ".join(words)


    def save(
        self,
        vocab_path: PathLike,
        merges_path: Optional[PathLike] = None,
    ) -> None:
        """
        Save tokenizer artifacts (vocab + merges).
        """
        if merges_path is None:
            raise ValueError("merges_path is required for BPE save.")

        self._save_vocab(vocab_path)
        self._save_merges(merges_path)


    @staticmethod
    def _build_initial_corpus(
        texts: Sequence[str],
    ) -> Tuple[List[List[Symbol]], set[Symbol]]:
        """
        Build:
          - corpus_words: list of [sym0, sym1, ..., sym_n] for each word
          - vocab: set of all symbols
        """
        corpus_words: List[List[Symbol]] = []
        vocab: set[Symbol] = set()

        for text in texts:
            for tok in _pretokenize(text):
                if not tok:
                    continue
                if tok in RESERVED_SPECIAL_TOKENS:
                    corpus_words.append([tok])
                    vocab.add(tok)
                    continue


                chars = list(tok)
                chars[-1] = chars[-1] + "</w>"  # attach end-of-word marker
                corpus_words.append(chars)
                vocab.update(chars)

        return corpus_words, vocab

    @staticmethod
    def _count_pairs(words: List[List[Symbol]]) -> Dict[Pair, int]:
        pair_counts: Dict[Pair, int] = {}
        for w in words:
            if len(w) < 2:
                continue
            for a, b in zip(w, w[1:]):
                pair = (a, b)
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts

    @staticmethod
    def _merge_pair_in_corpus(
        words: List[List[Symbol]],
        pair: Pair,
        new_symbol: Symbol,
    ) -> None:
        """
        In-place merge of 'pair' into 'new_symbol' for all words.
        """
        a, b = pair
        for i, w in enumerate(words):
            if len(w) < 2:
                continue
            words[i] = SubwordTokenizer._merge_pair_in_tokens(w, a, b, new_symbol)

    @staticmethod
    def _merge_pair_in_tokens(
        tokens: List[Symbol],
        a: Symbol,
        b: Symbol,
        new_symbol: Symbol,
    ) -> List[Symbol]:
        """
        Merge all (a, b) occurrences in a symbol sequence into new_symbol.
        """
        out: List[Symbol] = []
        i = 0
        last = len(tokens) - 1
        while i <= last:
            if i < last and tokens[i] == a and tokens[i + 1] == b:
                out.append(new_symbol)
                i += 2
            else:
                out.append(tokens[i])
                i += 1
        return out


    def _bpe_encode_word(self, word: str) -> List[Symbol]:
        """
        Apply BPE merges to a single word (no spaces).
        Returns a list of BPE symbols (strings).
        """
        if not word:
            return []

        tokens: List[Symbol] = list(word)
        tokens[-1] = tokens[-1] + "</w>"

        if not self.pair2rank or len(tokens) < 2:
            return tokens

        while True:
            # Build current adjacent pairs
            pairs: List[Pair] = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]

            # Find pair with smallest rank (earliest merge)
            best_pair: Optional[Pair] = None
            best_rank: Optional[int] = None
            for p in pairs:
                rank = self.pair2rank.get(p)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = p
                    best_rank = rank

            if best_pair is None:
                break

            a, b = best_pair
            new_symbol = a + b
            tokens = self._merge_pair_in_tokens(tokens, a, b, new_symbol)

            if len(tokens) < 2:
                break

        return tokens


    def _save_vocab(self, vocab_path: PathLike) -> None:
        vocab_path = Path(vocab_path)
        symbols = [self.itos[i] for i in range(len(self.itos))]
        with vocab_path.open("w", encoding="utf-8") as f:
            json.dump(symbols, f, ensure_ascii=False, indent=2)

    def _save_merges(self, merges_path: PathLike) -> None:
        merges_path = Path(merges_path)
        with merges_path.open("w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    @staticmethod
    def _load_vocab(vocab_path: PathLike) -> List[Symbol]:
        vocab_path = Path(vocab_path)
        with vocab_path.open("r", encoding="utf-8") as f:
            symbols = json.load(f)
        if not isinstance(symbols, list) or not all(isinstance(s, str) for s in symbols):
            raise ValueError("vocab.json must be a JSON list of strings (symbols in ID order).")
        return symbols

    @staticmethod
    def _load_merges(merges_path: PathLike) -> List[Pair]:
        merges_path = Path(merges_path)
        merges: List[Pair] = []
        with merges_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a, b = line.split()
                merges.append((a, b))
        return merges
