"""Runtime validation and resolution of immutable OC22 data profiles.

The profile artifacts and materialized LMDBs follow the contract published by
``irrep-dynamics/src/dataset/oc22``.  Cache construction intentionally remains
an offline data-preparation operation; training only resolves and validates an
already materialized cache.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lmdb
import numpy as np
import yaml


FORMAT_VERSION = 1
FULL_SELECTION_STRATEGY = "all_raw_samples"


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing required OC22 profile artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _lmdb_length(path: Path) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"Missing OC22 profile LMDB: {path}")
    env = lmdb.open(str(path), subdir=False, readonly=True, lock=False, readahead=False, max_readers=1)
    try:
        with env.begin() as txn:
            value = txn.get(b"length")
            if value is None:
                raise ValueError(f"OC22 profile LMDB has no length key: {path}")
            return int(pickle.loads(value))
    finally:
        env.close()


def resolve_profile_id(registry_path: Path, alias_or_id: str) -> str:
    registry = _load_yaml(registry_path)
    entry = registry.get("profiles", {}).get(alias_or_id)
    if entry is None:
        return alias_or_id
    profile_id = entry.get("id") if isinstance(entry, dict) else entry
    if not profile_id:
        raise ValueError(f"OC22 profile alias {alias_or_id!r} has no immutable ID")
    return str(profile_id)


def _selection_checksum(directory: Path, metadata: dict, filename: str) -> str | None:
    if metadata.get("selection", {}).get("strategy") == FULL_SELECTION_STRATEGY:
        return None
    path = directory / filename
    array = np.load(path)
    actual = _sha256_array(array)
    expected = metadata.get(f"{path.stem}_sha256")
    checksums_path = directory / "checksums.json"
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    if expected != actual or checksums.get(filename) != actual:
        raise ValueError(f"Invalid canonical OC22 indices artifact: {path}")
    return actual


@dataclass(frozen=True)
class ProfileCache:
    profile_id: str
    validation_id: str
    cache_dir: Path
    train_lmdb: Path
    validation_lmdb: Path
    train_natoms: np.ndarray
    validation_natoms: np.ndarray
    manifest: dict


def load_profile_cache(oc22_root: Path, registry_path: Path, alias_or_id: str) -> ProfileCache:
    """Resolve an alias and fail closed if its local cache is stale/incomplete."""
    oc22_root = Path(oc22_root)
    profile_id = resolve_profile_id(Path(registry_path), alias_or_id)
    profile_dir = oc22_root / "profiles" / profile_id
    metadata = _load_yaml(profile_dir / "metadata.yaml")
    if metadata.get("id") != profile_id:
        raise ValueError(f"OC22 profile metadata ID does not match {profile_id}")

    validation_id = str(metadata["validation_protocol_id"])
    validation_dir = oc22_root / "profiles" / "validation" / validation_id
    validation_metadata = _load_yaml(validation_dir / "metadata.yaml")
    if validation_metadata.get("id") != validation_id:
        raise ValueError(f"OC22 validation metadata ID does not match {validation_id}")

    train_checksum = _selection_checksum(profile_dir, metadata, "train_indices.npy")
    validation_checksum = _selection_checksum(validation_dir, validation_metadata, "indices.npy")
    cache_dir = oc22_root / "processed" / "profiles" / profile_id
    manifest = _load_yaml(cache_dir / "cache_manifest.yaml")
    required = {
        "format_version": FORMAT_VERSION,
        "profile_id": profile_id,
        "validation_protocol_id": validation_id,
        "source_manifest_sha256": metadata["source_manifest_sha256"],
        "validation_source_manifest_sha256": validation_metadata["source_manifest_sha256"],
    }
    required["train_indices_sha256" if train_checksum else "train_selection_strategy"] = (
        train_checksum or FULL_SELECTION_STRATEGY
    )
    required["validation_indices_sha256" if validation_checksum else "validation_selection_strategy"] = (
        validation_checksum or FULL_SELECTION_STRATEGY
    )
    mismatch = {key: (manifest.get(key), value) for key, value in required.items() if manifest.get(key) != value}
    if mismatch:
        raise ValueError(f"Stale or incompatible OC22 profile cache {cache_dir}: {mismatch}")

    validation_name = str(manifest.get("validation_cache_name", "val_proxy"))
    if validation_name not in {"val_proxy", "val_id"}:
        raise ValueError(f"Invalid OC22 validation cache name: {validation_name!r}")
    train_lmdb = cache_dir / "train.lmdb"
    validation_lmdb = cache_dir / f"{validation_name}.lmdb"
    train_natoms = np.load(cache_dir / "train_sample_natoms.npy")
    validation_natoms = np.load(cache_dir / f"{validation_name}_sample_natoms.npy")
    train_order = np.load(cache_dir / "train_local_to_raw.npy")
    validation_order = np.load(cache_dir / f"{validation_name}_local_to_raw.npy")

    for split, path, natoms, order, checksum_key in (
        ("train", train_lmdb, train_natoms, train_order, "train_order_sha256"),
        ("validation", validation_lmdb, validation_natoms, validation_order, "validation_order_sha256"),
    ):
        length = _lmdb_length(path)
        if natoms.shape != (length,) or order.shape != (length,):
            raise ValueError(f"OC22 {split} cache lengths disagree in {cache_dir}")
        if np.any(natoms <= 0) or manifest.get(checksum_key) != _sha256_array(order):
            raise ValueError(f"Invalid OC22 {split} cache metadata in {cache_dir}")

    return ProfileCache(
        profile_id=profile_id,
        validation_id=validation_id,
        cache_dir=cache_dir,
        train_lmdb=train_lmdb,
        validation_lmdb=validation_lmdb,
        train_natoms=train_natoms,
        validation_natoms=validation_natoms,
        manifest=manifest,
    )
