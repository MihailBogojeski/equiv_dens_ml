"""One-time Torch conversion; native JAX execution does not import this module.

Use this module in the project's Torch runtime. Arrays keep the original dense
weight orientation and shared tensor identities. No JAX import is required.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import hashlib
from pathlib import Path

import numpy as np
import torch

from .serialization import save_tree


def to_numpy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {to_numpy_tree(k): to_numpy_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_numpy_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy_tree(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.dtype):
        return str(value).split(".")[-1]
    if callable(value) and hasattr(value, "__name__"):
        return {"function": value.__name__, "module": value.__module__}
    raise TypeError(f"Cannot export metadata of type {type(value).__name__}")


def export_module(model):
    """Export every parameter/buffer and relevant plain module attribute."""
    state = {"params": {}, "buffers": {}}
    identities = {"params": {}, "buffers": {}}
    skipped = []

    def register(kind, path, value):
        identity = id(value)
        if identity not in identities[kind]:
            identities[kind][identity] = path
            state[kind][path] = to_numpy_tree(value)
        return identities[kind][identity]

    def visit(module, path):
        node = {"type": type(module).__name__, "path": path, "attrs": {},
                "children": {}, "params": {}, "buffers": {},"trainable":{}}
        for kind, members in (("params", module._parameters), ("buffers", module._buffers)):
            for name, value in members.items():
                if value is not None:
                    full = f"{path}.{name}" if path else name
                    node[kind][name] = register(kind, full, value)
                    if kind=="params":
                        node["trainable"][name]=bool(value.requires_grad)
        for name, value in vars(module).items():
            if name in {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"}:
                continue
            if name.startswith(("_forward", "_backward", "_state_dict", "_load_state_dict")):
                continue
            if isinstance(value, torch.nn.Module):
                continue
            try:
                node["attrs"][name] = to_numpy_tree(value)
            except TypeError:
                skipped.append({"path": path, "attribute": name, "type": type(value).__name__})
        for name, submodule in module._modules.items():
            if submodule is not None:
                full = f"{path}.{name}" if path else name
                node["children"][name] = visit(submodule, full)
        return node

    return visit(model, ""), state, skipped


def save_model(path, model, *, inputs=None, metadata=None):
    node, state, skipped = export_module(model)
    payload = {"model": node, "state": state,
               "metadata": {"torch": torch.__version__, "skipped_attributes": skipped,
                            **(metadata or {})}}
    if inputs is not None:
        payload["inputs"] = to_numpy_tree(inputs)
    return save_tree(path, payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert a current Torch DenSnet into portable JAX model arrays")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--factory", help="module:function returning the matching Torch model")
    parser.add_argument("--config", type=Path, help="JSON keyword arguments for the factory")
    parser.add_argument("--fixture", choices=("tiny", "batch", "wide"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = None
    if args.fixture:
        from benchmarks.fixtures import fixture_spec, build_model, make_atoms
        model = build_model(fixture_spec(args.fixture), "ef")
        inputs = make_atoms(args.fixture)
    elif args.factory:
        module_name, function_name = args.factory.split(":", 1)
        config = json.loads(args.config.read_text()) if args.config else {}
        model = getattr(importlib.import_module(module_name), function_name)(**config)
    elif args.checkpoint:
        model = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(model, torch.nn.Module):
            parser.error("State-only checkpoint requires --factory to construct its matching architecture")
    else:
        parser.error("Supply --fixture, --factory or a full-model --checkpoint")
    if args.checkpoint and (args.fixture or args.factory):
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(saved, torch.nn.Module):
            weights = saved.state_dict()
        else:
            weights = saved.get("model_state_dict", saved.get("state_dict", saved))
        model.load_state_dict(weights, strict=True)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    save_model(args.output, model.cpu(), inputs=inputs,
               metadata={"source_revision": revision, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                         "source_files":{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
                                         for path in sorted((root/"src/equiv_dens").rglob("*.py"))}})
    print(f"Exported {type(model).__name__} to {args.output}")


if __name__ == "__main__":
    main()
