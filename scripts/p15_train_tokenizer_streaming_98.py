from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    build_tokenizer_artifact_from_train_file,
)
from tinyllama_p15_prepare_finewebedu import (
    _canonicalize_doc,
    _is_train_doc,
    _iter_docs_from_source,
    DEFAULT_DATASET_ID,
    DEFAULT_DATASET_NAME,
    DEFAULT_TEXT_FIELD,
)


@dataclass
class Stream98Stats:
    seen_docs: int = 0
    non_empty_docs: int = 0
    train_docs: int = 0
    val_docs: int = 0
    train_bytes: int = 0
    val_bytes: int = 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stream FineWeb-Edu, keep deterministic 98%% train split, and train tokenizer on that full train slice."
    )
    p.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    p.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME)
    p.add_argument("--text-field", type=str, default=DEFAULT_TEXT_FIELD)
    p.add_argument("--source-file", type=Path, default=None, help="Optional local source file; bypasses HF streaming.")
    p.add_argument("--train-ratio", type=float, default=0.98)
    p.add_argument("--split-seed", type=int, default=1337)
    p.add_argument("--vocab-size", type=int, default=16000)
    p.add_argument("--backend-family", type=str, default="sentencepiece", choices=("sentencepiece",))
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--max-docs", type=int, default=None, help="Optional cap for smoke/dev runs.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(
        "[stream98] start "
        f"dataset={args.dataset_id}/{args.dataset_name} "
        f"train_ratio={args.train_ratio} split_seed={args.split_seed} "
        f"vocab_size={args.vocab_size} output_root={output_root}",
        flush=True,
    )

    split_dir = output_root / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_file = split_dir / "train_98.txt"
    val_file = split_dir / "valid_2.txt"

    stats = Stream98Stats()
    source_file = args.source_file.expanduser().resolve() if args.source_file is not None else None
    if source_file is not None and not source_file.exists():
        raise FileNotFoundError(source_file)

    with train_file.open("w", encoding="utf-8") as train_f, val_file.open("w", encoding="utf-8") as val_f:
        for raw in _iter_docs_from_source(
            source_file=source_file,
            dataset_id=str(args.dataset_id),
            dataset_name=str(args.dataset_name),
            text_field=str(args.text_field),
        ):
            stats.seen_docs += 1
            doc = _canonicalize_doc(raw)
            if not doc:
                continue
            if args.max_docs is not None and stats.non_empty_docs >= int(args.max_docs):
                break
            stats.non_empty_docs += 1
            idx = stats.non_empty_docs - 1
            doc_bytes = len(doc.encode("utf-8")) + 1
            if _is_train_doc(
                split_seed=int(args.split_seed),
                train_ratio=float(args.train_ratio),
                index=idx,
                doc=doc,
            ):
                train_f.write(doc + "\n")
                stats.train_docs += 1
                stats.train_bytes += doc_bytes
            else:
                val_f.write(doc + "\n")
                stats.val_docs += 1
                stats.val_bytes += doc_bytes
            if (stats.non_empty_docs % 100000) == 0:
                print(
                    "[stream98] progress "
                    f"non_empty_docs={stats.non_empty_docs} "
                    f"train_docs={stats.train_docs} val_docs={stats.val_docs} "
                    f"train_gb={stats.train_bytes / (1024**3):.3f} val_gb={stats.val_bytes / (1024**3):.3f}",
                    flush=True,
                )

    if stats.train_docs <= 0:
        raise RuntimeError("no train docs written; check source/split settings")
    print(
        "[stream98] split_complete "
        f"non_empty_docs={stats.non_empty_docs} train_docs={stats.train_docs} val_docs={stats.val_docs}",
        flush=True,
    )

    tokenizer_dir = output_root / "tokenizer"
    build_tokenizer_artifact_from_train_file(
        input_file=train_file,
        output_dir=tokenizer_dir,
        vocab_size=int(args.vocab_size),
        model_type=str(args.backend_family),  # type: ignore[arg-type]
    )
    print("[stream98] tokenizer_training_complete", flush=True)

    tok_hash = (tokenizer_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).read_text(encoding="utf-8").strip()
    report = {
        "dataset": {
            "dataset_id": str(args.dataset_id),
            "dataset_name": str(args.dataset_name),
            "text_field": str(args.text_field),
            "source_file": None if source_file is None else str(source_file),
        },
        "split": {
            "algorithm": "tinyllama_p15_sha256(seed|index|doc_utf8) < train_ratio",
            "train_ratio": float(args.train_ratio),
            "split_seed": int(args.split_seed),
            "train_file": str(train_file),
            "val_file": str(val_file),
        },
        "tokenizer": {
            "backend_family": str(args.backend_family),
            "vocab_size": int(args.vocab_size),
            "tokenizer_dir": str(tokenizer_dir),
            "tokenizer_hash": tok_hash,
        },
        "stats": asdict(stats),
    }
    report_path = output_root / "run_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"run_report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
