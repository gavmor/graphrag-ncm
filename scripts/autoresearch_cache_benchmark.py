"""Offline cache acceptance workload.

Candidate cache contract: ncm_graphrag.ncm.generator_cache.load_or_build_cache(
    vqvae, checkpoint_path, prefix_data, target_data, cache_path)
returns a mapping with `idx` and `rest_embed` tensors. The loader must reject or
rebuild stale/legacy files. This adapter is deliberately separate from production.
"""

import importlib.util
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from models.next_concept import NextConceptTransformer
from models.vqvae import VQVAE

from ncm_graphrag.ncm.config import build_model_cfg, build_tokenizer_cfg


def train_step(vq, config, prefix, idx, rest):
    net = NextConceptTransformer(vqvae_local=vq, config=config)
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    with torch.no_grad():
        prefix_embed = vq.encoder.wte(prefix)
    inputs = torch.cat((prefix_embed, net.build_l0_query_input(len(prefix), dtype=rest.dtype), rest), dim=1)
    logits = net(inputs)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), idx.reshape(-1))
    if not torch.isfinite(loss):
        raise ValueError("non-finite generator loss")
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def main():
    torch.manual_seed(1729)
    torch.set_num_threads(1)
    cfg = build_tokenizer_cfg().vqvae
    cfg.in_vocab_size = 64
    cfg.context_length = 8
    cfg.embed_dim = 16
    cfg.n_layers = 2
    cfg.vqvae_vocab_size = [8, 8, 8]
    cfg.vector_quantizer_config.levels.manual_levels = [2, 4, 8]
    cfg.dropout = 0.0
    cfg.pre_quant_dropout = 0.0
    vq = VQVAE(**cfg).eval()
    for p in vq.parameters():
        p.requires_grad_(False)
    model_cfg = build_model_cfg()
    model_cfg.n_layer = 1
    model_cfg.n_head = 2
    model_cfg.n_embd = 16
    model_cfg.vocab_size = [8, 8, 8]
    model_cfg.prefix_len = 4
    model_cfg.dropout = 0.0
    prefix = torch.randint(0, 64, (6, 4))
    target = torch.randint(0, 64, (6, 8))
    with torch.no_grad():
        expected_idx = vq.encode(target)
        expected_rest = vq.vq.idx_to_ncm_input(expected_idx)
    # Always exercise the real VQ-VAE and generator, even before a cache exists.
    train_step(vq, model_cfg, prefix, expected_idx, expected_rest)

    scores = dict(
        cache_recorded=0,
        cache_hit=0,
        checkpoint_change=0,
        dataset_change=0,
        legacy_rejected=0,
        cached_values=0,
        cached_generator=0,
    )
    module = "ncm_graphrag.ncm.generator_cache"
    if importlib.util.find_spec(module) is not None:
        load_or_build_cache = importlib.import_module(module).load_or_build_cache

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "tokenizer.pt"
            cache = root / "cache.pt"
            torch.save({"model": vq.state_dict()}, checkpoint)

            def load(p=prefix, t=target):
                return load_or_build_cache(vq, checkpoint, p, t, cache)

            first = load()
            scores["cache_recorded"] = int(cache.is_file())
            # An actual hit must avoid recomputing codes.
            original_encode = vq.encode

            def no_encode(*args, **kwargs):
                raise AssertionError("cache hit recomputed codes")

            setattr(vq, "encode", no_encode)
            try:
                second = load()
                scores["cache_hit"] = 1
            except AssertionError as exc:
                if str(exc) != "cache hit recomputed codes":
                    raise
                second = first
            finally:
                setattr(vq, "encode", original_encode)
            scores["cached_values"] = int(
                torch.equal(second["idx"], expected_idx)
                and torch.allclose(second["rest_embed"], expected_rest, rtol=1e-5, atol=1e-6)
            )
            if scores["cache_hit"] and scores["cached_values"]:
                train_step(vq, model_cfg, prefix, second["idx"], second["rest_embed"])
                scores["cached_generator"] = 1

            def changed_ok(p, t):
                calls = []

                def counted_encode(*args, **kwargs):
                    calls.append(1)
                    return original_encode(*args, **kwargs)

                setattr(vq, "encode", counted_encode)
                try:
                    try:
                        result = load(p, t)
                    except (ValueError, RuntimeError):
                        return True
                finally:
                    setattr(vq, "encode", original_encode)
                fresh = vq.encode(t)
                return (
                    bool(calls)
                    and torch.equal(result["idx"], fresh)
                    and torch.allclose(result["rest_embed"], vq.vq.idx_to_ncm_input(fresh), rtol=1e-5, atol=1e-6)
                )

            # Change checkpoint content and model weights at the same path.
            with torch.no_grad():
                vq.encoder.wte.weight.add_(torch.randn_like(vq.encoder.wte.weight) * 2)
            torch.save({"model": vq.state_dict()}, checkpoint)
            scores["checkpoint_change"] = int(changed_ok(prefix, target))
            # Re-establish a matching baseline if the previous change was rejected.
            if not cache.exists():
                load()
            else:
                try:
                    load()
                except (ValueError, RuntimeError):
                    cache.unlink()
                    load()
            changed_target = target.roll(1, 0)
            changed_prefix = prefix.roll(1, 0)
            scores["dataset_change"] = int(changed_ok(changed_prefix, changed_target))
            torch.save({"idx": expected_idx, "rest_embed": expected_rest}, cache)
            scores["legacy_rejected"] = int(changed_ok(prefix, target))
    else:
        print("No generator cache implementation found; baseline scores zero.")
    for name, value in scores.items():
        print(f"METRIC {name}={value}")
    print(f"METRIC acceptance={sum(scores.values())}")


if __name__ == "__main__":
    main()
