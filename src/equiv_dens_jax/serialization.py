"""Portable JSON/NPZ trees, including tuple keys; no pickle or Torch dependency."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

SCHEMA = 1


def _encode(value, arrays):
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "__array__")
    ):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot be stored in a portable bundle")
        key = f"array_{len(arrays):06d}"
        arrays[key] = array
        return {"array": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, dict):
        return {"dict": [[_encode(k, arrays), _encode(v, arrays)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {"tuple": [_encode(v, arrays) for v in value]}
    if isinstance(value, list):
        return {"list": [_encode(v, arrays) for v in value]}
    if isinstance(value, float) and not math.isfinite(value):
        return {"float": str(value)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported portable tree leaf: {type(value).__name__}")


def _decode(value, arrays):
    if not isinstance(value, dict):
        return value
    if set(value) == {"array"}:
        return arrays[value["array"]].copy()
    if set(value) == {"dict"}:
        return {_decode(k, arrays): _decode(v, arrays) for k, v in value["dict"]}
    if set(value) == {"tuple"}:
        return tuple(_decode(v, arrays) for v in value["tuple"])
    if set(value) == {"list"}:
        return [_decode(v, arrays) for v in value["list"]]
    if set(value) == {"float"}:
        return float(value["float"])
    raise ValueError("Invalid portable tree encoding")


def save_tree(path, tree):
    """Atomically write one bundle directory, committing its manifest last."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {}
    encoded = _encode(tree, arrays)
    with tempfile.TemporaryDirectory(prefix=".bundle-", dir=path) as temporary:
        array_path = Path(temporary) / "arrays.npz"
        np.savez_compressed(array_path, **arrays)
        digest = hashlib.sha256(array_path.read_bytes()).hexdigest()
        # Content-address the arrays so concurrent readers of the old manifest
        # cannot accidentally see new arrays with the old tree description.
        array_name = f"arrays-{digest}.npz"
        manifest = {"schema": SCHEMA, "arrays": array_name, "sha256": digest, "tree": encoded}
        manifest_path = Path(temporary) / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        os.replace(array_path, path / array_name)
        os.replace(manifest_path, path / "manifest.json")
    return path


def load_tree(path):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported bundle schema: {manifest.get('schema')}")
    name = manifest["arrays"]
    if Path(name).name != name:
        raise ValueError("Array archive must be a local bundle file")
    array_path = path / name
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Bundle array archive checksum mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        return _decode(manifest["tree"], archive)
