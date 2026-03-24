from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm_lab.core.data.pretok_shards import (
    ROOT_MANIFEST_FILENAME,
    PRETOK_FORMAT_VERSION,
    TinyLlamaP15DocSpan,
    TinyLlamaP15ShardSidecar,
    _apply_eot_insertion_policy,
    _apply_normalization_policy,
    _resolve_split_membership_semantics,
    _summarize_split_counts,
    build_tinyllama_p15_pretokenized_shards,
    build_prep_config_ref,
    chunk_split_token_stream,
    compute_docs_sha256,
    deterministic_shard_basename,
    read_json,
    read_split_manifest,
    read_uint16_tokens,
    sha256_hex_bytes,
    sha256_hex_file,
    spot_decode_shard,
    validate_split_manifest_structure,
    validate_strict_provenance,
    write_shard_sidecar,
    write_uint16_tokens,
)
from llm_lab.core.tokenization.subword_tokenizer import SubwordTokenizer, SubwordTokenizerConfig
from llm_lab.core.tokenization.tinyllama_p15_tokenizer_artifact import (
    TOKENIZER_ARTIFACT_FILENAMES,
    build_tokenizer_artifact_from_docs,
    compute_tokenizer_hash,
)


# Function map (supporting contracts -> test)
# - split-manifest schema/provenance checks: test_split_manifest_structure_and_provenance_checks
# - deterministic shard naming/chunking: test_deterministic_shard_naming_and_chunking
# - uint16 overflow refusal + roundtrip: test_uint16_overflow_refusal_and_roundtrip
# - checksum helpers determinism: test_checksum_stability
# - sidecar schema write/read: test_sidecar_schema_write_read
# - spot decode helper + CLI validation: test_spot_decode_helper, test_cli_validation_build_mode_missing_inputs
# - normalization/split/EOT/count semantics + build E2E: tests at end of this file


def _script_path() -> Path:
    return REPO_ROOT / "scripts" / "tinyllama_p15_pretokenize_shards.py"


def test_split_manifest_structure_and_provenance_checks() -> None:
    manifest = {
        "contract_version": "tinyllama_p15_tokenizer_artifact_v1",
        "split_seed": 7,
        "total_docs": 3,
        "train_doc_indices": [0, 2],
        "val_doc_indices": [1],
        "docs_sha256": "abc",
        "tokenizer_hash": "tokhash",
    }
    validate_split_manifest_structure(manifest)

    validate_strict_provenance(
        manifest,
        tokenizer_hash="tokhash",
        docs_sha256="abc",
        doc_count=3,
    )

    bad = dict(manifest)
    bad.pop("tokenizer_hash")
    with pytest.raises(ValueError):
        validate_split_manifest_structure(bad)

    with pytest.raises(ValueError, match="tokenizer hash mismatch"):
        validate_strict_provenance(manifest, tokenizer_hash="other", docs_sha256="abc", doc_count=3)


def test_deterministic_shard_naming_and_chunking() -> None:
    assert deterministic_shard_basename("train", 0) == "train_000000"
    assert deterministic_shard_basename("val", 12) == "val_000012"

    token_stream = list(range(10))
    spans = [
        TinyLlamaP15DocSpan(global_doc_index=0, start_token=0, end_token=3),
        TinyLlamaP15DocSpan(global_doc_index=1, start_token=3, end_token=7),
        TinyLlamaP15DocSpan(global_doc_index=2, start_token=7, end_token=10),
    ]
    chunks = chunk_split_token_stream(
        split="train",
        token_stream=token_stream,
        doc_spans=spans,
        max_tokens_per_shard=4,
    )

    assert [c.shard_index for c in chunks] == [0, 1, 2]
    assert [c.token_ids for c in chunks] == [list(range(4)), list(range(4, 8)), list(range(8, 10))]
    assert chunks[0].first_doc_global_index == 0
    assert chunks[0].last_doc_global_index == 1


def test_uint16_overflow_refusal_and_roundtrip(tmp_path: Path) -> None:
    shard_path = tmp_path / "ok.bin"
    write_uint16_tokens(shard_path, [0, 1, 42, 65535])
    assert read_uint16_tokens(shard_path) == [0, 1, 42, 65535]

    with pytest.raises(ValueError, match="cannot be represented as uint16"):
        write_uint16_tokens(tmp_path / "overflow.bin", [0, 70000])


def test_checksum_stability(tmp_path: Path) -> None:
    payload = b"tinyllama-p15"
    h1 = sha256_hex_bytes(payload)
    h2 = sha256_hex_bytes(payload)
    assert h1 == h2

    f = tmp_path / "x.bin"
    f.write_bytes(payload)
    assert sha256_hex_file(f) == sha256_hex_bytes(payload)


def test_sidecar_schema_write_read(tmp_path: Path) -> None:
    sidecar = TinyLlamaP15ShardSidecar(
        format_version=PRETOK_FORMAT_VERSION,
        filename="train_000000.bin",
        split="train",
        dtype="uint16",
        shard_index=0,
        token_count=10,
        document_count=2,
        tokenizer_hash="tokhash",
        split_manifest_sha256="splithash",
        sha256="shardhash",
        first_doc_global_index=0,
        last_doc_global_index=1,
        prep_config_ref=build_prep_config_ref({"a": 1}),
    )
    path = tmp_path / "train_000000.json"
    write_shard_sidecar(path, sidecar)
    loaded = read_json(path)

    required = {
        "format_version",
        "filename",
        "split",
        "dtype",
        "shard_index",
        "token_count",
        "document_count",
        "tokenizer_hash",
        "split_manifest_sha256",
        "sha256",
        "first_doc_global_index",
        "last_doc_global_index",
        "prep_config_ref",
    }
    assert required.issubset(set(loaded.keys()))


def test_spot_decode_helper(tmp_path: Path) -> None:
    tok = SubwordTokenizer.train_from_iterator(
        ["alpha beta", "beta alpha"],
        SubwordTokenizerConfig(vocab_size=80, model_type="bpe"),
    )
    eot_id = tok.token_to_id("<|endoftext|>")

    ids = tok.encode("alpha beta") + [eot_id] + tok.encode("beta alpha") + [eot_id]
    shard = tmp_path / "train_000000.bin"
    write_uint16_tokens(shard, ids)

    decoded = spot_decode_shard(shard, tokenizer=tok, eot_token_id=eot_id, max_docs=2)
    assert len(decoded) == 2
    assert decoded[0]["token_count"] > 0


def test_cli_validation_build_mode_missing_inputs(tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        str(_script_path()),
        "--tokenizer-artifact-dir",
        str(tmp_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "build mode requires" in proc.stderr


def test_cli_spot_decode_mode(tmp_path: Path) -> None:
    tok = SubwordTokenizer.train_from_iterator(
        ["alpha beta", "beta alpha"],
        SubwordTokenizerConfig(vocab_size=80, model_type="bpe"),
    )
    tok_dir = tmp_path / "tok"
    tok_dir.mkdir(parents=True)
    tok.save(
        tok_dir / TOKENIZER_ARTIFACT_FILENAMES["vocab"],
        tok_dir / TOKENIZER_ARTIFACT_FILENAMES["merges"],
    )
    (tok_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).write_text(
        compute_tokenizer_hash(tok) + "\n",
        encoding="utf-8",
    )

    eot_id = tok.token_to_id("<|endoftext|>")
    ids = tok.encode("alpha beta") + [eot_id]
    shard = tmp_path / "train_000000.bin"
    write_uint16_tokens(shard, ids)

    cmd = [
        sys.executable,
        str(_script_path()),
        "--tokenizer-artifact-dir",
        str(tok_dir),
        "--spot-decode-shard",
        str(shard),
        "--spot-decode-max-docs",
        "1",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert isinstance(out, list)
    assert len(out) == 1


def test_read_split_manifest_from_file(tmp_path: Path) -> None:
    manifest = {
        "contract_version": "tinyllama_p15_tokenizer_artifact_v1",
        "split_seed": 7,
        "total_docs": 3,
        "train_doc_indices": [0, 2],
        "val_doc_indices": [1],
        "docs_sha256": compute_docs_sha256(["a", "b", "c"]),
        "tokenizer_hash": "tokhash",
    }
    path = tmp_path / "split_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = read_split_manifest(path)
    assert loaded["total_docs"] == 3


def test_apply_normalization_policy_identity() -> None:
    docs = ["alpha  beta", "gamma\tdelta", "epsilon"]
    normalized = _apply_normalization_policy(docs)
    assert normalized == docs


def test_resolve_split_membership_semantics_validates_coverage() -> None:
    manifest = {
        "train_doc_indices": [0, 2],
        "val_doc_indices": [1],
        "total_docs": 3,
    }
    resolved = _resolve_split_membership_semantics(manifest, normalized_doc_count=3)
    assert resolved["train"] == [0, 2]
    assert resolved["val"] == [1]

    with pytest.raises(ValueError, match="cover all normalized docs"):
        _resolve_split_membership_semantics(
            {
                "train_doc_indices": [0],
                "val_doc_indices": [1],
                "total_docs": 3,
            },
            normalized_doc_count=3,
        )


def test_apply_eot_insertion_policy_appends_final_eot() -> None:
    stream, spans = _apply_eot_insertion_policy(
        [(0, [11, 12]), (4, []), (8, [99])],
        eot_id=3,
    )
    assert stream == [11, 12, 3, 3, 99, 3]
    assert spans == [
        TinyLlamaP15DocSpan(global_doc_index=0, start_token=0, end_token=3),
        TinyLlamaP15DocSpan(global_doc_index=4, start_token=3, end_token=4),
        TinyLlamaP15DocSpan(global_doc_index=8, start_token=4, end_token=6),
    ]


def test_summarize_split_counts_includes_eot_tokens() -> None:
    docs = ["a", "b"]
    stream = [10, 3, 11, 3]
    spans = [
        TinyLlamaP15DocSpan(global_doc_index=0, start_token=0, end_token=2),
        TinyLlamaP15DocSpan(global_doc_index=1, start_token=2, end_token=4),
    ]
    doc_count, token_count = _summarize_split_counts(
        split_docs=docs,
        split_token_stream=stream,
        doc_spans=spans,
    )
    assert doc_count == 2
    assert token_count == 4

    with pytest.raises(ValueError, match="doc span count mismatch"):
        _summarize_split_counts(split_docs=docs, split_token_stream=stream, doc_spans=spans[:1])


def test_build_tinyllama_p15_pretokenized_shards_e2e(tmp_path: Path) -> None:
    # Keep a shared symbol inventory across docs so val docs remain encodable with a train-only tokenizer.
    docs = ["ab ab", "ba ab", "aa bb", "bb aa"]
    input_file = tmp_path / "docs.txt"
    input_file.write_text("\n".join(docs) + "\n", encoding="utf-8")

    tok_dir = tmp_path / "tok"
    build_tokenizer_artifact_from_docs(
        docs=docs,
        output_dir=tok_dir,
        vocab_size=80,
        train_ratio=0.75,
        split_seed=2,
    )
    split_manifest_path = tok_dir / TOKENIZER_ARTIFACT_FILENAMES["split_manifest"]
    split_manifest = read_json(split_manifest_path)
    assert split_manifest["tokenizer_hash"] == (
        (tok_dir / TOKENIZER_ARTIFACT_FILENAMES["tokenizer_hash"]).read_text(encoding="utf-8").strip()
    )

    out_dir = tmp_path / "shards"
    root_manifest_path = build_tinyllama_p15_pretokenized_shards(
        input_file=input_file,
        tokenizer_artifact_dir=tok_dir,
        split_manifest_path=split_manifest_path,
        output_dir=out_dir,
        max_tokens_per_shard=8,
    )

    assert root_manifest_path == out_dir / ROOT_MANIFEST_FILENAME
    root = read_json(root_manifest_path)
    assert root["format_version"] == PRETOK_FORMAT_VERSION
    assert root["normalization_policy_marker"] == "tinyllama_p15_norm_v1_identity_preserve_docs"
    assert root["eot_policy_marker"] == "tinyllama_p15_eot_v1_append_after_every_doc_including_final"
    assert root["total_docs_per_split"]["train"] + root["total_docs_per_split"]["val"] == len(docs)
    assert root["total_tokens_per_split"]["train"] >= root["total_docs_per_split"]["train"]
    assert root["total_tokens_per_split"]["val"] >= root["total_docs_per_split"]["val"]
    assert root["shard_inventory"]

    for entry in root["shard_inventory"]:
        sidecar = read_json(out_dir / entry["sidecar"])
        assert sidecar["format_version"] == PRETOK_FORMAT_VERSION
        assert sidecar["dtype"] == "uint16"
        assert sidecar["token_count"] >= sidecar["document_count"]
