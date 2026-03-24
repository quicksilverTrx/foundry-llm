from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Union

from llm_lab.core.tokenization.legacy_bpe_backend import LegacyBPEBackend
from llm_lab.core.tokenization.sentencepiece_backend import SentencePieceBackend
from llm_lab.core.tokenization.tokenizer_shared import (
    ID2SPECIAL,
    RESERVED_SPECIAL_TOKENS,
    get_pretokenizer_spec,
    pretokenize,
)

PathLike = Union[str, Path]
BackendFamily = Literal["legacy_bpe", "sentencepiece"]
__all__ = [
    "RESERVED_SPECIAL_TOKENS",
    "ID2SPECIAL",
    "get_pretokenizer_spec",
    "SubwordTokenizerConfig",
    "SubwordTokenizer",
    "_pretokenize",
]


def _normalize_model_type(model_type: str) -> BackendFamily:
    if model_type == "bpe":
        return "legacy_bpe"
    if model_type == "legacy_bpe":
        return "legacy_bpe"
    if model_type == "sentencepiece":
        return "sentencepiece"
    raise ValueError(f"Unsupported tokenizer backend family: {model_type}")


# Backward-compat export for tests and existing imports.
def _pretokenize(text: str) -> List[str]:
    return pretokenize(text)


@dataclass
class SubwordTokenizerConfig:
    vocab_size: int
    model_type: Literal["bpe", "legacy_bpe", "sentencepiece"] = "bpe"
    sentencepiece_model_type: Literal["bpe", "unigram"] = "bpe"
    sentencepiece_character_coverage: float = 1.0

    @property
    def backend_family(self) -> BackendFamily:
        return _normalize_model_type(self.model_type)


class SubwordTokenizer:
    """
    Stable tokenizer facade that delegates to backend implementations.
    """

    _FILENAME_VOCAB = "vocab.json"
    _FILENAME_MERGES = "merges.txt"
    _FILENAME_SENTENCEPIECE_MODEL = "sentencepiece.model"
    _FILENAME_EXTERNAL_ID_MAP = "external_id_map.json"
    _FILENAME_SENTENCEPIECE_META = "sentencepiece_meta.json"
    _FILENAME_TOKENIZER_CONFIG = "tokenizer_config.json"

    def __init__(
        self,
        *,
        backend_impl: LegacyBPEBackend | SentencePieceBackend,
        config: SubwordTokenizerConfig,
        backend_family: BackendFamily,
        backend_metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        self._backend = backend_impl
        self.config = config
        self.backend_family: BackendFamily = backend_family
        self._backend_metadata: Dict[str, object] = dict(backend_metadata or {})
        self._refresh_compat_views()

    def _refresh_compat_views(self) -> None:
        # Compatibility surfaces retained for existing callers.
        self.stoi = dict(self._backend.stoi)
        self.itos = dict(self._backend.itos)
        self.merges = list(getattr(self._backend, "merges", []))

    @property
    def backend_metadata(self) -> Dict[str, object]:
        out = dict(self._backend_metadata)
        out["backend_family"] = self.backend_family
        return out

    def token_to_id(self, tok: str) -> int:
        return self._backend.token_to_id(tok)

    def id_to_token(self, idx: int) -> str:
        return self._backend.id_to_token(idx)

    def is_special(self, tok: str) -> bool:
        return tok in RESERVED_SPECIAL_TOKENS

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,  # kept for API compatibility
    ) -> List[int]:
        del add_special_tokens
        return self._backend.encode(text)

    def decode(self, ids: Sequence[int]) -> str:
        return self._backend.decode(ids)

    def save(
        self,
        vocab_path: PathLike | None = None,
        merges_path: PathLike | None = None,
        *,
        artifact_dir: PathLike | None = None,
    ) -> None:
        if artifact_dir is not None:
            output_dir = Path(artifact_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            if self.backend_family == "legacy_bpe":
                legacy_backend = self._require_backend(LegacyBPEBackend)
                legacy_backend.save(
                    vocab_path=output_dir / self._FILENAME_VOCAB,
                    merges_path=output_dir / self._FILENAME_MERGES,
                )
                return
            sp_backend = self._require_backend(SentencePieceBackend)
            sp_backend.save(
                sentencepiece_model_path=output_dir / self._FILENAME_SENTENCEPIECE_MODEL,
                external_id_map_path=output_dir / self._FILENAME_EXTERNAL_ID_MAP,
                sentencepiece_meta_path=output_dir / self._FILENAME_SENTENCEPIECE_META,
            )
            return

        if self.backend_family == "legacy_bpe":
            if vocab_path is None or merges_path is None:
                raise ValueError("legacy_bpe save requires both vocab_path and merges_path")
            legacy_backend = self._require_backend(LegacyBPEBackend)
            legacy_backend.save(vocab_path=vocab_path, merges_path=merges_path)
            return

        if vocab_path is None:
            raise ValueError("sentencepiece save requires vocab_path or artifact_dir")
        model_path = Path(vocab_path)
        external_map_path = (
            Path(merges_path)
            if merges_path is not None
            else model_path.with_name(self._FILENAME_EXTERNAL_ID_MAP)
        )
        meta_path = model_path.with_name(self._FILENAME_SENTENCEPIECE_META)
        sp_backend = self._require_backend(SentencePieceBackend)
        sp_backend.save(
            sentencepiece_model_path=model_path,
            external_id_map_path=external_map_path,
            sentencepiece_meta_path=meta_path,
        )

    @classmethod
    def train_from_iterator(
        cls,
        texts: Sequence[str],
        config: SubwordTokenizerConfig,
    ) -> "SubwordTokenizer":
        family = config.backend_family
        if family == "legacy_bpe":
            backend = LegacyBPEBackend.train_from_iterator(
                texts,
                vocab_size=config.vocab_size,
                reserved_tokens=RESERVED_SPECIAL_TOKENS,
                pretokenize_fn=pretokenize,
            )
            return cls(
                backend_impl=backend,
                config=config,
                backend_family=family,
                backend_metadata={"legacy_bpe_merge_count": len(backend.merges)},
            )

        backend = SentencePieceBackend.train_from_iterator(
            texts,
            vocab_size=config.vocab_size,
            reserved_tokens=RESERVED_SPECIAL_TOKENS,
            model_type=config.sentencepiece_model_type,
            character_coverage=config.sentencepiece_character_coverage,
        )
        return cls(
            backend_impl=backend,
            config=config,
            backend_family=family,
            backend_metadata=backend.sentencepiece_meta,
        )

    @classmethod
    def train_from_file(
        cls,
        input_file: PathLike,
        config: SubwordTokenizerConfig,
    ) -> "SubwordTokenizer":
        family = config.backend_family
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        if family == "legacy_bpe":
            texts = [line.strip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return cls.train_from_iterator(texts=texts, config=config)

        backend = SentencePieceBackend.train_from_file(
            input_file=input_path,
            vocab_size=config.vocab_size,
            reserved_tokens=RESERVED_SPECIAL_TOKENS,
            model_type=config.sentencepiece_model_type,
            character_coverage=config.sentencepiece_character_coverage,
        )
        return cls(
            backend_impl=backend,
            config=config,
            backend_family="sentencepiece",
            backend_metadata=backend.sentencepiece_meta,
        )

    @classmethod
    def load_from_files(
        cls,
        vocab_path: PathLike,
        merges_path: Optional[PathLike] = None,
        config: Optional[SubwordTokenizerConfig] = None,
    ) -> "SubwordTokenizer":
        if merges_path is None:
            raise ValueError("merges_path is required for legacy_bpe load")
        backend = LegacyBPEBackend.load_from_files(
            vocab_path=vocab_path,
            merges_path=merges_path,
            reserved_tokens=RESERVED_SPECIAL_TOKENS,
            pretokenize_fn=pretokenize,
        )
        cfg = config or SubwordTokenizerConfig(
            vocab_size=len(backend.stoi),
            model_type="legacy_bpe",
        )
        return cls(
            backend_impl=backend,
            config=cfg,
            backend_family="legacy_bpe",
            backend_metadata={"legacy_bpe_merge_count": len(backend.merges)},
        )

    @classmethod
    def load(cls, artifact_dir: PathLike) -> "SubwordTokenizer":
        tok_dir = Path(artifact_dir)
        cfg_path = tok_dir / cls._FILENAME_TOKENIZER_CONFIG

        backend_hint = None
        if cfg_path.exists():
            import json

            cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8"))
            backend_hint = cfg_obj.get("backend_family") or cfg_obj.get("model_type")

        if backend_hint is None:
            if (tok_dir / cls._FILENAME_SENTENCEPIECE_MODEL).exists():
                backend_hint = "sentencepiece"
            elif (tok_dir / cls._FILENAME_VOCAB).exists() and (tok_dir / cls._FILENAME_MERGES).exists():
                backend_hint = "legacy_bpe"
            else:
                raise FileNotFoundError(f"Cannot infer tokenizer backend from artifact dir: {tok_dir}")

        family = _normalize_model_type(str(backend_hint))
        if family == "legacy_bpe":
            return cls.load_from_files(
                vocab_path=tok_dir / cls._FILENAME_VOCAB,
                merges_path=tok_dir / cls._FILENAME_MERGES,
            )

        sp_backend = SentencePieceBackend.load_from_files(
            sentencepiece_model_path=tok_dir / cls._FILENAME_SENTENCEPIECE_MODEL,
            external_id_map_path=tok_dir / cls._FILENAME_EXTERNAL_ID_MAP,
            sentencepiece_meta_path=tok_dir / cls._FILENAME_SENTENCEPIECE_META,
            reserved_tokens=RESERVED_SPECIAL_TOKENS,
        )
        cfg = SubwordTokenizerConfig(
            vocab_size=len(sp_backend.stoi),
            model_type="sentencepiece",
        )
        return cls(
            backend_impl=sp_backend,
            config=cfg,
            backend_family="sentencepiece",
            backend_metadata=sp_backend.sentencepiece_meta,
        )

    def export_backend_payload(self, output_dir: PathLike) -> Dict[str, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.backend_family == "legacy_bpe":
            self.save(
                vocab_path=out / self._FILENAME_VOCAB,
                merges_path=out / self._FILENAME_MERGES,
            )
            return {
                "vocab": out / self._FILENAME_VOCAB,
                "merges": out / self._FILENAME_MERGES,
            }

        self.save(artifact_dir=out)
        return {
            "sentencepiece_model": out / self._FILENAME_SENTENCEPIECE_MODEL,
            "external_id_map": out / self._FILENAME_EXTERNAL_ID_MAP,
            "sentencepiece_meta": out / self._FILENAME_SENTENCEPIECE_META,
        }

    def ordered_vocab_symbols(self) -> List[str]:
        return [self.itos[i] for i in sorted(self.itos.keys())]

    def iter_backend_hash_components(self) -> Iterable[object]:
        if self.backend_family == "legacy_bpe":
            yield {"vocab_symbols_in_id_order": self.ordered_vocab_symbols()}
            yield {"merges_in_rank_order": [[a, b] for a, b in self.merges]}
            return

        sp_backend = self._require_backend(SentencePieceBackend)
        yield {"vocab_symbols_in_id_order": self.ordered_vocab_symbols()}
        yield {"sentencepiece_behavior_fingerprint": sp_backend.behavior_fingerprint()}
        yield {"external_id_map": sp_backend.build_external_id_map_payload()}
        yield {"sentencepiece_meta": sp_backend.sentencepiece_meta}

    def _require_backend(self, backend_cls: type[LegacyBPEBackend] | type[SentencePieceBackend]):
        if isinstance(self._backend, backend_cls):
            return self._backend
        raise TypeError(f"Tokenizer backend type mismatch: expected {backend_cls.__name__}")
