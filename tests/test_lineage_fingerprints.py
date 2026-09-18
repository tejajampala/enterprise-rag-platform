from __future__ import annotations

from consumer_finance_rag.lineage.fingerprints import (
    ManagedSnapshotFingerprintRecord,
    chunking_fingerprint,
    embedding_fingerprint,
    managed_snapshot_hash,
    metadata_hash,
    version_hash,
)


def snapshot_record(source_key: str, marker: str) -> ManagedSnapshotFingerprintRecord:
    return ManagedSnapshotFingerprintRecord(
        source_key=source_key,
        content_hash=marker * 64,
        metadata_hash=marker.upper() * 64,
        version_hash=("f" if marker != "f" else "e") * 64,
    )


def test_managed_snapshot_hash_is_deterministic_and_order_independent() -> None:
    first = managed_snapshot_hash([snapshot_record("b", "b"), snapshot_record("a", "a")])
    second = managed_snapshot_hash([snapshot_record("a", "a"), snapshot_record("b", "b")])

    assert first == second
    assert len(first) == 64


def test_managed_snapshot_hash_changes_with_source_version_identity() -> None:
    original = managed_snapshot_hash([snapshot_record("a", "a")])
    changed = managed_snapshot_hash([snapshot_record("a", "b")])

    assert original != changed


def metadata_fingerprint(*, title: str = "Credit reports", source_url: str | None = None) -> str:
    return metadata_hash(
        title=title,
        source_url=source_url,
        category="credit-reporting",
        source_version="2026-09-18",
    )


def test_metadata_hash_is_deterministic_full_sha256() -> None:
    first = metadata_fingerprint(source_url="https://example.test/reports")
    second = metadata_fingerprint(source_url="https://example.test/reports")

    assert first == second
    assert len(first) == 64
    assert set(first) <= set("0123456789abcdef")


def test_metadata_hash_changes_with_title() -> None:
    assert metadata_fingerprint(title="Credit reports") != metadata_fingerprint(
        title="Understanding credit reports"
    )


def test_metadata_hash_changes_with_source_url() -> None:
    assert metadata_fingerprint(source_url="https://example.test/one") != metadata_fingerprint(
        source_url="https://example.test/two"
    )


def test_metadata_hash_distinguishes_none_from_empty_string() -> None:
    assert metadata_fingerprint(source_url=None) != metadata_fingerprint(source_url="")


def test_version_hash_is_deterministic() -> None:
    first = version_hash(content_hash="a" * 64, metadata_hash="b" * 64)
    second = version_hash(content_hash="a" * 64, metadata_hash="b" * 64)

    assert first == second
    assert len(first) == 64


def test_version_hash_changes_with_content_hash() -> None:
    assert version_hash(content_hash="a" * 64, metadata_hash="b" * 64) != version_hash(
        content_hash="c" * 64, metadata_hash="b" * 64
    )


def test_version_hash_changes_with_metadata_hash() -> None:
    assert version_hash(content_hash="a" * 64, metadata_hash="b" * 64) != version_hash(
        content_hash="a" * 64, metadata_hash="c" * 64
    )


def test_chunking_fingerprint_is_deterministic() -> None:
    assert chunking_fingerprint(chunk_size=900, overlap=120) == chunking_fingerprint(
        overlap=120, chunk_size=900
    )


def test_chunking_fingerprint_changes_with_chunk_size() -> None:
    assert chunking_fingerprint(chunk_size=900, overlap=120) != chunking_fingerprint(
        chunk_size=1000, overlap=120
    )


def test_embedding_fingerprint_is_deterministic_and_device_independent() -> None:
    first = embedding_fingerprint(
        provider="huggingface",
        model="Qwen/Qwen3-Embedding-0.6B",
        device="cpu",
    )
    second = embedding_fingerprint(
        model="Qwen/Qwen3-Embedding-0.6B",
        provider="huggingface",
        device="cuda",
    )

    assert first == second
    assert len(first) == 64


def test_embedding_fingerprint_changes_with_document_encoding_contract() -> None:
    default = embedding_fingerprint(
        provider="huggingface", model="Qwen/Qwen3-Embedding-0.6B"
    )
    changed = embedding_fingerprint(
        provider="huggingface",
        model="Qwen/Qwen3-Embedding-0.6B",
        document_encoding_mode="encode",
    )

    assert default != changed
