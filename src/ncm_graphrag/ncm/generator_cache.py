"""Content-addressed provenance for frozen VQ-VAE training tensors.

Stale and pre-provenance caches fail closed; remove the named file to rebuild.
Only load cache/checkpoint files from trusted local training runs.
"""

import hashlib
import os
import tempfile
from pathlib import Path

import torch
from models.vqvae import VQVAE


def _tensor_identity(tensor: torch.Tensor) -> str:
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
    digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_identity(state) -> dict[str, str]:
    return {name: _tensor_identity(value) for name, value in sorted(state.items())}


@torch.no_grad()
def load_or_build_cache(
    vqvae: VQVAE,
    checkpoint_path: str | Path,
    prefix_data: torch.Tensor,
    target_data: torch.Tensor,
    cache_path: str | Path,
) -> dict[str, torch.Tensor]:
    """Return CPU codes and embeddings bound to exact checkpoint and token rows."""
    path = Path(cache_path)
    if any(module.training for module in vqvae.modules()):
        raise ValueError("Generator caching requires a VQ-VAE in eval mode")
    if prefix_data.ndim != 2 or target_data.ndim != 2 or len(prefix_data) != len(target_data) or not len(target_data):
        raise ValueError("Cache inputs must be nonempty, row-aligned token matrices")
    if prefix_data.dtype != torch.long or target_data.dtype != torch.long:
        raise ValueError("Cache inputs must contain int64 token IDs")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_digest = hashlib.sha256()
    with checkpoint_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checkpoint_digest.update(chunk)
    checkpoint_hash = checkpoint_digest.hexdigest()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_identity = _state_identity(vqvae.state_dict())
    if model_identity != _state_identity(checkpoint["model"]):
        raise ValueError("Loaded VQ-VAE weights do not match tokenizer checkpoint")
    provenance = {
        "format_version": 1,
        "checkpoint_sha256": checkpoint_hash,
        "model_state": model_identity,
        "architecture": repr(vqvae),
        "levels": list(vqvae.vq.levels),
        "prefix_sha256": _tensor_identity(prefix_data),
        "target_sha256": _tensor_identity(target_data),
    }

    def invalid(reason):
        return ValueError(f"Invalid generator cache {path}: {reason}. Remove this file and rerun to rebuild.")

    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("provenance") != provenance:
            raise invalid("checkpoint, tokenized dataset, or cache format changed (legacy caches are unsupported)")
        data = {name: payload.get(name) for name in ("idx", "rest_embed")}
        levels = list(vqvae.vq.levels)
        idx, rest = data["idx"], data["rest_embed"]
        if not isinstance(idx, torch.Tensor) or not isinstance(rest, torch.Tensor):
            raise invalid("missing cached tensors")
        if idx.dtype != torch.long or idx.shape != (len(target_data), sum(levels)):
            raise invalid("code shape or dtype mismatch")
        if rest.shape != (len(target_data), sum(levels[1:]), vqvae.embed_dim):
            raise invalid("embedding shape mismatch")
        if payload.get("tensor_sha256") != {name: _tensor_identity(value) for name, value in data.items()}:
            raise invalid("cached tensor contents changed")
        return data

    device = next(vqvae.parameters()).device
    codes, embeddings = [], []
    for start in range(0, len(target_data), 64):
        idx = vqvae.encode(target_data[start : start + 64].to(device))
        codes.append(idx.cpu())
        embeddings.append(vqvae.vq.idx_to_ncm_input(idx).cpu())
    data = {"idx": torch.cat(codes), "rest_embed": torch.cat(embeddings)}
    payload = {
        "provenance": provenance,
        **data,
        "tensor_sha256": {name: _tensor_identity(value) for name, value in data.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement prevents a killed writer leaving a seemingly valid cache.
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return data
