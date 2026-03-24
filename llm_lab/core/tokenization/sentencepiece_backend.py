from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Union

PathLike = Union[str, Path]

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - validated in tests that need sentencepiece
    raise ImportError(
        "sentencepiece is required for sentencepiece tokenizer backend. "
        "Install dependency `sentencepiece` in foundry-llm environment."
    ) from exc


class SentencePieceBackend:
    backend_family = "sentencepiece"
    EXTERNAL_MAP_VERSION = "tinyllama_p15_sentencepiece_external_map_v1"

    def __init__(
        self,
        *,
        processor: "spm.SentencePieceProcessor",
        reserved_tokens: Dict[str, int],
        external_to_internal: Dict[int, int],
        sentencepiece_meta: Dict[str, object] | None = None,
    ) -> None:
        self._sp = processor
        self._reserved_tokens = dict(reserved_tokens)
        self._validate_external_map_contract(
            external_to_internal=external_to_internal,
            reserved_tokens=self._reserved_tokens,
            piece_count=self._sp.get_piece_size(),
        )
        self._external_to_internal = dict(external_to_internal)
        self._internal_to_external = {v: k for k, v in self._external_to_internal.items()}
        self._sentencepiece_meta = dict(sentencepiece_meta or {})

        self.stoi: Dict[str, int] = dict(self._reserved_tokens)
        self.itos: Dict[int, str] = {idx: tok for tok, idx in self._reserved_tokens.items()}
        self.merges: List[tuple[str, str]] = []

        for internal_id in range(self._sp.get_piece_size()):
            ext_id = self._internal_to_external.get(internal_id)
            if ext_id is None:
                continue
            piece = self._sp.id_to_piece(internal_id)
            if piece in self._reserved_tokens:
                continue
            self.stoi[piece] = ext_id
            self.itos[ext_id] = piece

    @classmethod
    def train_from_iterator(
        cls,
        texts: Sequence[str],
        *,
        vocab_size: int,
        reserved_tokens: Dict[str, int],
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> "SentencePieceBackend":
        reserved_count = len(reserved_tokens)
        internal_vocab_size = max(16, vocab_size - reserved_count)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            train_path = td_path / "sp_train.txt"
            train_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
            model_prefix = td_path / "spm"

            spm.SentencePieceTrainer.train(
                input=str(train_path),
                model_prefix=str(model_prefix),
                vocab_size=internal_vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                normalization_rule_name="identity",
                input_sentence_size=0,
                shuffle_input_sentence=False,
                num_threads=1,
                hard_vocab_limit=False,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )

            processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

        external_to_internal = cls._resolve_external_internal_id_map(
            processor=processor, reserved_tokens=reserved_tokens
        )
        sentencepiece_meta = {
            "backend_family": cls.backend_family,
            "internal_model_type": model_type,
            "internal_piece_count": processor.get_piece_size(),
            "character_coverage": character_coverage,
            "normalization_rule_name": "identity",
        }
        return cls(
            processor=processor,
            reserved_tokens=reserved_tokens,
            external_to_internal=external_to_internal,
            sentencepiece_meta=sentencepiece_meta,
        )

    @classmethod
    def train_from_file(
        cls,
        input_file: PathLike,
        *,
        vocab_size: int,
        reserved_tokens: Dict[str, int],
        model_type: str = "bpe",
        character_coverage: float = 1.0,
    ) -> "SentencePieceBackend":
        reserved_count = len(reserved_tokens)
        internal_vocab_size = max(16, vocab_size - reserved_count)
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_prefix = td_path / "spm"

            spm.SentencePieceTrainer.train(
                input=str(input_path),
                model_prefix=str(model_prefix),
                vocab_size=internal_vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                normalization_rule_name="identity",
                input_sentence_size=0,
                shuffle_input_sentence=False,
                num_threads=1,
                hard_vocab_limit=False,
                unk_id=0,
                bos_id=-1,
                eos_id=-1,
                pad_id=-1,
            )

            processor = spm.SentencePieceProcessor(model_file=str(model_prefix) + ".model")

        external_to_internal = cls._resolve_external_internal_id_map(
            processor=processor, reserved_tokens=reserved_tokens
        )
        sentencepiece_meta = {
            "backend_family": cls.backend_family,
            "internal_model_type": model_type,
            "internal_piece_count": processor.get_piece_size(),
            "character_coverage": character_coverage,
            "normalization_rule_name": "identity",
            "training_source": str(input_path),
        }
        return cls(
            processor=processor,
            reserved_tokens=reserved_tokens,
            external_to_internal=external_to_internal,
            sentencepiece_meta=sentencepiece_meta,
        )

    @classmethod
    def load_from_files(
        cls,
        *,
        sentencepiece_model_path: PathLike,
        external_id_map_path: PathLike | None,
        sentencepiece_meta_path: PathLike | None,
        reserved_tokens: Dict[str, int],
    ) -> "SentencePieceBackend":
        processor = spm.SentencePieceProcessor(model_file=str(sentencepiece_model_path))
        if external_id_map_path is not None and Path(external_id_map_path).exists():
            payload = json.loads(Path(external_id_map_path).read_text(encoding="utf-8"))
            external_to_internal = cls._external_map_from_payload(
                payload=payload,
                reserved_tokens=reserved_tokens,
                piece_count=processor.get_piece_size(),
            )
        else:
            external_to_internal = cls._resolve_external_internal_id_map(
                processor=processor, reserved_tokens=reserved_tokens
            )

        sentencepiece_meta: Dict[str, object] = {}
        if sentencepiece_meta_path is not None and Path(sentencepiece_meta_path).exists():
            sentencepiece_meta = json.loads(Path(sentencepiece_meta_path).read_text(encoding="utf-8"))

        return cls(
            processor=processor,
            reserved_tokens=reserved_tokens,
            external_to_internal=external_to_internal,
            sentencepiece_meta=sentencepiece_meta,
        )

    def save(
        self,
        *,
        sentencepiece_model_path: PathLike,
        external_id_map_path: PathLike,
        sentencepiece_meta_path: PathLike,
    ) -> None:
        model_path = Path(sentencepiece_model_path)
        model_path.write_bytes(self._sp.serialized_model_proto())

        external_payload = self.build_external_id_map_payload()
        Path(external_id_map_path).write_text(
            json.dumps(external_payload, sort_keys=True, indent=2), encoding="utf-8"
        )
        Path(sentencepiece_meta_path).write_text(
            json.dumps(self._sentencepiece_meta, sort_keys=True, indent=2), encoding="utf-8"
        )

    def token_to_id(self, token: str) -> int:
        return self.stoi[token]

    def id_to_token(self, idx: int) -> str:
        return self.itos[idx]

    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        cursor = 0
        reserved_literals = tuple(self._reserved_tokens.keys())

        # Only exact reserved literals are special. Unknown/malformed "<|...|>" spans
        # stay on the ordinary non-reserved text path.
        while cursor < len(text):
            next_pos = None
            next_literal = None
            for literal in reserved_literals:
                pos = text.find(literal, cursor)
                if pos == -1:
                    continue
                if next_pos is None or pos < next_pos:
                    next_pos = pos
                    next_literal = literal

            if next_pos is None:
                out.extend(self._encode_non_special_chunk(text[cursor:]))
                break

            if next_pos > cursor:
                out.extend(self._encode_non_special_chunk(text[cursor:next_pos]))

            out.append(self._reserved_tokens[next_literal])
            cursor = next_pos + len(next_literal)

        return out

    def decode(self, ids: Sequence[int]) -> str:
        parts: List[str] = []
        internal_buf: List[int] = []
        reserved_ids = set(self._reserved_tokens.values())

        def flush_non_special() -> None:
            if internal_buf:
                parts.append(self._sp.decode(internal_buf))
                internal_buf.clear()

        for external_id in ids:
            if external_id in reserved_ids:
                flush_non_special()
                parts.append(self.id_to_token(external_id))
                continue
            internal_buf.append(self._external_to_internal[external_id])

        flush_non_special()
        return "".join(parts)

    def build_external_id_map_payload(self) -> Dict[str, object]:
        return {
            "contract_version": self.EXTERNAL_MAP_VERSION,
            "backend_family": self.backend_family,
            "reserved_tokens": self._reserved_tokens,
            "external_to_internal": {
                str(external): internal
                for external, internal in sorted(self._external_to_internal.items())
            },
            "internal_piece_count": self._sp.get_piece_size(),
        }

    def model_sha256(self) -> str:
        import hashlib

        return hashlib.sha256(self._sp.serialized_model_proto()).hexdigest()

    def behavior_fingerprint(self) -> Dict[str, object]:
        return {
            "internal_piece_count": self._sp.get_piece_size(),
            "pieces_in_id_order": [self._sp.id_to_piece(i) for i in range(self._sp.get_piece_size())],
            "piece_scores_in_id_order": [
                self._sp.get_score(i) for i in range(self._sp.get_piece_size())
            ],
        }

    @property
    def sentencepiece_meta(self) -> Dict[str, object]:
        return dict(self._sentencepiece_meta)

    def _encode_non_special_chunk(self, text: str) -> List[int]:
        internal_ids = self._sp.encode(text, out_type=int)
        return [self._internal_to_external[internal_id] for internal_id in internal_ids]

    @classmethod
    def _external_map_from_payload(
        cls,
        *,
        payload: Dict[str, object],
        reserved_tokens: Dict[str, int],
        piece_count: int,
    ) -> Dict[int, int]:
        raw = payload.get("external_to_internal")
        if not isinstance(raw, dict):
            raise ValueError("external_id_map.json missing external_to_internal mapping")
        parsed = {int(k): int(v) for k, v in raw.items()}
        cls._validate_external_map_contract(
            external_to_internal=parsed,
            reserved_tokens=reserved_tokens,
            piece_count=piece_count,
        )
        return parsed

    @classmethod
    def _resolve_external_internal_id_map(
        cls,
        *,
        processor: "spm.SentencePieceProcessor",
        reserved_tokens: Dict[str, int],
    ) -> Dict[int, int]:
        reserved_ids = set(reserved_tokens.values())
        piece_count = processor.get_piece_size()

        start_external = max(reserved_ids) + 1
        external_to_internal = {
            start_external + internal_id: internal_id for internal_id in range(piece_count)
        }
        cls._validate_external_map_contract(
            external_to_internal=external_to_internal,
            reserved_tokens=reserved_tokens,
            piece_count=piece_count,
        )
        return external_to_internal

    @classmethod
    def _validate_external_map_contract(
        cls,
        *,
        external_to_internal: Dict[int, int],
        reserved_tokens: Dict[str, int],
        piece_count: int,
    ) -> None:
        reserved_ids = set(reserved_tokens.values())
        for ext_id in external_to_internal:
            if ext_id in reserved_ids:
                raise ValueError("external_id_map.json must not map reserved external IDs")

        start_external = max(reserved_ids) + 1
        expected_external = set(range(start_external, start_external + piece_count))
        actual_external = set(external_to_internal.keys())
        if actual_external != expected_external:
            raise ValueError("external_id_map.json external IDs must be contiguous from reserved block end")

        values = list(external_to_internal.values())
        if len(values) != len(set(values)):
            raise ValueError("external_id_map.json internal IDs must be unique")
        expected_internal = set(range(piece_count))
        if set(values) != expected_internal:
            raise ValueError("external_id_map.json internal IDs must match sentencepiece piece ID domain")
