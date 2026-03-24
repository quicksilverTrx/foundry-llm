from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# TinyLlama bridge START: tinyllama_p15 tokenizer artifact tests
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.tokenization.subword_tokenizer import (
    RESERVED_SPECIAL_TOKENS,
    SubwordTokenizer,
    SubwordTokenizerConfig,
)
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    TOKENIZER_CONTRACT_VERSION,
    build_tokenizer_artifact_from_docs,
    load_docs_one_per_non_empty_line,
    load_tokenizer_from_artifact_dir,
)


def _repo_root() -> Path:
    return REPO_ROOT


def _script_path() -> Path:
    return _repo_root() / "scripts" / "tinyllama_p15_train_tokenizer.py"


def _run_cli(
    *,
    input_file: Path,
    output_dir: Path,
    vocab_size: int,
    backend_family: str = "sentencepiece",
    train_ratio: float = 0.98,
    split_seed: int = 0,
) -> None:
    cmd = [
        sys.executable,
        str(_script_path()),
        "--input-file",
        str(input_file),
        "--output-dir",
        str(output_dir),
        "--vocab-size",
        str(vocab_size),
        "--backend-family",
        backend_family,
        "--train-ratio",
        str(train_ratio),
        "--split-seed",
        str(split_seed),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_cli_sentencepiece_build_is_deterministic_for_same_inputs(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text(
        "\n".join(
            [
                "alpha beta gamma",
                "beta gamma alpha",
                "alpha alpha beta",
                "gamma alpha beta",
                "beta alpha gamma",
                "alpha beta alpha",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out1 = tmp_path / "tok_a"
    out2 = tmp_path / "tok_b"
    _run_cli(
        input_file=corpus,
        output_dir=out1,
        vocab_size=120,
        backend_family="sentencepiece",
        train_ratio=0.75,
        split_seed=7,
    )
    _run_cli(
        input_file=corpus,
        output_dir=out2,
        vocab_size=120,
        backend_family="sentencepiece",
        train_ratio=0.75,
        split_seed=7,
    )

    assert (out1 / "external_id_map.json").read_text(encoding="utf-8") == (
        out2 / "external_id_map.json"
    ).read_text(encoding="utf-8")
    assert (out1 / "tokenizer_hash.txt").read_text(encoding="utf-8") == (
        out2 / "tokenizer_hash.txt"
    ).read_text(encoding="utf-8")


def test_sentencepiece_artifact_metadata_is_present_and_parseable(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text("hello world\nhello there\nworld hello\n", encoding="utf-8")
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=80,
        backend_family="sentencepiece",
        train_ratio=0.8,
        split_seed=11,
    )

    common_required = ("config", "stats", "reserved_tokens", "split_manifest", "tokenizer_hash")
    sentencepiece_required = ("sentencepiece_model", "external_id_map", "sentencepiece_meta")
    for key in common_required + sentencepiece_required:
        assert (out / TOKENIZER_ARTIFACT_FILENAMES[key]).exists(), f"missing artifact file: {key}"

    config = json.loads((out / "tokenizer_config.json").read_text(encoding="utf-8"))
    stats = json.loads((out / "tokenizer_stats.json").read_text(encoding="utf-8"))
    reserved = json.loads((out / "reserved_tokens.json").read_text(encoding="utf-8"))
    split_manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    tokenizer_hash = (out / "tokenizer_hash.txt").read_text(encoding="utf-8").strip()

    assert config["contract_version"] == TOKENIZER_CONTRACT_VERSION
    assert stats["contract_version"] == TOKENIZER_CONTRACT_VERSION
    assert split_manifest["contract_version"] == TOKENIZER_CONTRACT_VERSION
    assert config["backend_family"] == "sentencepiece"
    assert reserved == RESERVED_SPECIAL_TOKENS
    assert config["doc_shape_policy"] == "one_non_empty_line_per_doc"
    assert split_manifest["tokenizer_hash"] == tokenizer_hash


def test_sentencepiece_external_map_contract_is_contiguous_and_bijective(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text("hello world\nhello there\nworld hello\n", encoding="utf-8")
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=80,
        backend_family="sentencepiece",
        train_ratio=0.8,
        split_seed=11,
    )

    payload = json.loads((out / "external_id_map.json").read_text(encoding="utf-8"))
    ext_map = {int(k): int(v) for k, v in payload["external_to_internal"].items()}
    piece_count = int(payload["internal_piece_count"])

    expected_external = list(range(4, 4 + piece_count))
    assert sorted(ext_map.keys()) == expected_external
    assert sorted(ext_map.values()) == list(range(piece_count))
    assert len(ext_map.values()) == len(set(ext_map.values()))
    assert set(ext_map.keys()).isdisjoint({0, 1, 2, 3})


def test_sentencepiece_external_map_rejects_reserved_collision(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text("hello world\nhello there\nworld hello\n", encoding="utf-8")
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=80,
        backend_family="sentencepiece",
        train_ratio=0.8,
        split_seed=11,
    )

    payload = json.loads((out / "external_id_map.json").read_text(encoding="utf-8"))
    payload["external_to_internal"]["1"] = 0
    (out / "external_id_map.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved external IDs"):
        load_tokenizer_from_artifact_dir(out)


def test_sentencepiece_external_map_rejects_non_contiguous_external_ids(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text("hello world\nhello there\nworld hello\n", encoding="utf-8")
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=80,
        backend_family="sentencepiece",
        train_ratio=0.8,
        split_seed=11,
    )

    payload = json.loads((out / "external_id_map.json").read_text(encoding="utf-8"))
    ext_map = payload["external_to_internal"]
    keys = sorted(ext_map.keys(), key=int)
    removed = keys[len(keys) // 2]
    ext_map.pop(removed)
    payload["external_to_internal"] = ext_map
    payload["internal_piece_count"] = payload["internal_piece_count"] - 1
    (out / "external_id_map.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous"):
        load_tokenizer_from_artifact_dir(out)


def test_legacy_bpe_artifact_regression_and_loader_compat(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text("hello world\nhello there\nworld hello\n", encoding="utf-8")
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=80,
        backend_family="legacy_bpe",
        train_ratio=0.8,
        split_seed=11,
    )

    assert (out / "vocab.json").exists()
    assert (out / "merges.txt").exists()
    tok, _ = load_tokenizer_from_artifact_dir(out)
    assert tok.backend_family == "legacy_bpe"
    assert tok.encode("hello world")


def test_split_before_training_is_enforced_for_legacy_bpe(tmp_path: Path) -> None:
    corpus = tmp_path / "docs.txt"
    corpus.write_text(
        "alpha beta alpha\nbeta gamma beta\ngamma alpha gamma\nalpha gamma alpha\n",
        encoding="utf-8",
    )
    out = tmp_path / "tok"
    _run_cli(
        input_file=corpus,
        output_dir=out,
        vocab_size=90,
        backend_family="legacy_bpe",
        train_ratio=0.5,
        split_seed=3,
    )

    docs = load_docs_one_per_non_empty_line(corpus)
    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    train_docs = [docs[idx] for idx in manifest["train_doc_indices"]]

    expected = SubwordTokenizer.train_from_iterator(
        train_docs, SubwordTokenizerConfig(vocab_size=90, model_type="legacy_bpe")
    )
    actual = SubwordTokenizer.load_from_files(out / "vocab.json", out / "merges.txt")

    assert actual.stoi == expected.stoi
    assert actual.merges == expected.merges


def test_backward_compat_loader_for_v1_like_legacy_payload(tmp_path: Path) -> None:
    tok = SubwordTokenizer.train_from_iterator(
        ["alpha beta", "beta alpha"],
        SubwordTokenizerConfig(vocab_size=80, model_type="legacy_bpe"),
    )
    out = tmp_path / "tok"
    out.mkdir(parents=True, exist_ok=True)
    tok.save(out / "vocab.json", out / "merges.txt")
    (out / "tokenizer_hash.txt").write_text("fakehash\n", encoding="utf-8")

    loaded, tok_hash = load_tokenizer_from_artifact_dir(out)
    assert loaded.backend_family == "legacy_bpe"
    assert tok_hash == "fakehash"


def test_invalid_train_ratio_raises_value_error() -> None:
    with pytest.raises(ValueError, match="train_ratio must be in \\(0, 1\\)"):
        build_tokenizer_artifact_from_docs(
            docs=["alpha beta", "beta gamma"],
            output_dir=Path("unused"),
            vocab_size=64,
            train_ratio=1.0,
            split_seed=0,
        )


def test_empty_train_split_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train split is empty"):
        build_tokenizer_artifact_from_docs(
            docs=["alpha beta", "beta gamma"],
            output_dir=tmp_path / "out",
            vocab_size=64,
            train_ratio=0.5,
            split_seed=0,
            split_score_fn=lambda _seed, _idx, _doc: 0.99999999,
        )


# TinyLlama bridge END: tinyllama_p15 tokenizer artifact tests
