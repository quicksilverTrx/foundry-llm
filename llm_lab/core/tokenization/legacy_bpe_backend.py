from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

Symbol = str
Pair = Tuple[Symbol, Symbol]
PathLike = Union[str, Path]
PretokenizeFn = Callable[[str], List[str]]


class LegacyBPEBackend:
    """
    Legacy char-level BPE backend retained for tinyllama_p15 compatibility.
    """

    backend_family = "legacy_bpe"

    def __init__(
        self,
        *,
        stoi: Dict[Symbol, int],
        itos: Dict[int, Symbol],
        merges: Optional[Sequence[Pair]] = None,
        reserved_tokens: Dict[str, int],
        pretokenize_fn: PretokenizeFn,
    ) -> None:
        self.stoi = stoi
        self.itos = itos
        self.merges: List[Pair] = list(merges) if merges is not None else []
        self.pair2rank: Dict[Pair, int] = {pair: i for i, pair in enumerate(self.merges)}
        self._reserved_tokens = reserved_tokens
        self._pretokenize = pretokenize_fn

    @classmethod
    def train_from_iterator(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int,
        reserved_tokens: Dict[str, int],
        pretokenize_fn: PretokenizeFn,
    ) -> "LegacyBPEBackend":
        corpus_words, vocab = cls._build_initial_corpus(
            texts, reserved_tokens=reserved_tokens, pretokenize_fn=pretokenize_fn
        )

        merges: List[Pair] = []
        while len(vocab) < vocab_size:
            pair_counts = cls._count_pairs(corpus_words)
            if not pair_counts:
                break

            best_pair, best_count = max(pair_counts.items(), key=lambda kv: kv[1])
            if best_count < 2:
                break

            merges.append(best_pair)
            new_symbol = best_pair[0] + best_pair[1]
            vocab.add(new_symbol)
            cls._merge_pair_in_corpus(corpus_words, best_pair, new_symbol)

        reserved_items = sorted(reserved_tokens.items(), key=lambda kv: kv[1])
        reserved_syms = [s for s, _ in reserved_items]
        learned_vocab = sorted(sym for sym in vocab if sym not in reserved_tokens)
        symbols = reserved_syms + learned_vocab
        for s, idx in reserved_tokens.items():
            assert symbols[idx] == s, (s, idx, symbols[idx])

        stoi = {sym: i for i, sym in enumerate(symbols)}
        itos = {i: sym for sym, i in stoi.items()}
        return cls(
            stoi=stoi,
            itos=itos,
            merges=merges,
            reserved_tokens=reserved_tokens,
            pretokenize_fn=pretokenize_fn,
        )

    @classmethod
    def load_from_files(
        cls,
        *,
        vocab_path: PathLike,
        merges_path: PathLike,
        reserved_tokens: Dict[str, int],
        pretokenize_fn: PretokenizeFn,
    ) -> "LegacyBPEBackend":
        symbols = cls._load_vocab(vocab_path)
        merges = cls._load_merges(merges_path)

        stoi = {sym: i for i, sym in enumerate(symbols)}
        itos = {i: sym for i, sym in enumerate(symbols)}
        return cls(
            stoi=stoi,
            itos=itos,
            merges=merges,
            reserved_tokens=reserved_tokens,
            pretokenize_fn=pretokenize_fn,
        )

    def token_to_id(self, tok: str) -> int:
        return self.stoi[tok]

    def id_to_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in self._pretokenize(text):
            if tok in self._reserved_tokens:
                ids.append(self.stoi[tok])
                continue

            for sym in self._bpe_encode_word(tok):
                try:
                    ids.append(self.stoi[sym])
                except KeyError:
                    raise KeyError(f"Unknown BPE symbol {sym!r}") from None
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        symbols = (self.itos[i] for i in ids)
        words: List[str] = []
        current = ""

        for sym in symbols:
            if sym in self._reserved_tokens:
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

    def save(self, *, vocab_path: PathLike, merges_path: PathLike) -> None:
        self._save_vocab(vocab_path)
        self._save_merges(merges_path)

    @staticmethod
    def _build_initial_corpus(
        texts: Sequence[str],
        *,
        reserved_tokens: Dict[str, int],
        pretokenize_fn: PretokenizeFn,
    ) -> Tuple[List[List[Symbol]], set[Symbol]]:
        corpus_words: List[List[Symbol]] = []
        vocab: set[Symbol] = set()

        for text in texts:
            for tok in pretokenize_fn(text):
                if not tok:
                    continue
                if tok in reserved_tokens:
                    corpus_words.append([tok])
                    vocab.add(tok)
                    continue

                chars = list(tok)
                chars[-1] = chars[-1] + "</w>"
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
        a, b = pair
        for i, w in enumerate(words):
            if len(w) < 2:
                continue
            words[i] = LegacyBPEBackend._merge_pair_in_tokens(w, a, b, new_symbol)

    @staticmethod
    def _merge_pair_in_tokens(
        tokens: List[Symbol],
        a: Symbol,
        b: Symbol,
        new_symbol: Symbol,
    ) -> List[Symbol]:
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
        if not word:
            return []

        tokens: List[Symbol] = list(word)
        tokens[-1] = tokens[-1] + "</w>"

        if not self.pair2rank or len(tokens) < 2:
            return tokens

        while True:
            pairs: List[Pair] = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
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
        text = vocab_path.read_text(encoding="utf-8")
        s = text.lstrip()

        if s.startswith("["):
            symbols = json.loads(text)
            if not isinstance(symbols, list) or not all(isinstance(x, str) for x in symbols):
                raise ValueError("vocab.json must be a JSON list of strings (symbols in ID order).")
            return symbols

        symbols = [ln.rstrip("\n") for ln in text.splitlines()]
        symbols = [t for t in symbols if t != ""]
        if not symbols:
            raise ValueError(f"Empty vocab file: {vocab_path}")
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
