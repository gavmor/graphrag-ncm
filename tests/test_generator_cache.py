"""Cache identity boundaries using the real multiscale VQ-VAE."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from models.vqvae import VQVAE

from ncm_graphrag.ncm.config import build_tokenizer_cfg
from ncm_graphrag.ncm.generator_cache import load_or_build_cache
from scripts.fewnerd_probe import _validate_cached_values, validate_cache


def tiny_tokenizer_cfg():
    cfg = build_tokenizer_cfg()
    cfg.vqvae.in_vocab_size = 64
    cfg.vqvae.context_length = 8
    cfg.vqvae.embed_dim = 16
    cfg.vqvae.n_layers = 2
    cfg.vqvae.vqvae_vocab_size = [8, 8, 8]
    cfg.vqvae.vector_quantizer_config.levels.manual_levels = [2, 4, 8]
    return cfg


class GeneratorCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.checkpoint = root / "tokenizer.pt"
        self.cache = root / "cache.pt"
        with torch.random.fork_rng():
            torch.manual_seed(32)
            self.vq = VQVAE(**tiny_tokenizer_cfg().vqvae).eval()
            self.prefix = torch.randint(0, 64, (5, 4))
            self.target = torch.randint(0, 64, (5, 8))
        torch.save({"model": self.vq.state_dict()}, self.checkpoint)

    def load(self, prefix=None, target=None):
        return load_or_build_cache(
            self.vq,
            self.checkpoint,
            self.prefix if prefix is None else prefix,
            self.target if target is None else target,
            self.cache,
        )

    def test_disk_roundtrip_matches_fresh_outputs(self):
        self.load()
        cached = self.load()
        idx = self.vq.encode(self.target)
        torch.testing.assert_close(cached["idx"], idx, rtol=0, atol=0)
        torch.testing.assert_close(cached["rest_embed"], self.vq.vq.idx_to_ncm_input(idx))

    def test_prefix_and_target_changes_independently_reject_same_shape_cache(self):
        self.load()
        for field in ("prefix", "target"):
            with self.subTest(field=field):
                changed = getattr(self, field).clone()
                changed[-1, -1] = (changed[-1, -1] + 1) % 64
                with self.assertRaisesRegex(ValueError, "Remove this file"):
                    self.load(**{field: changed})

    def test_checkpoint_replaced_at_same_path_is_rejected(self):
        self.load()
        with torch.no_grad():
            self.vq.encoder.wte.weight[0, 0].add_(1)
        torch.save({"model": self.vq.state_dict()}, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.load()

    def test_in_memory_model_must_match_checkpoint(self):
        with torch.no_grad():
            self.vq.encoder.wte.weight[0, 0].add_(1)
        with self.assertRaisesRegex(ValueError, "weights do not match"):
            self.load()

    def test_legacy_and_modified_payloads_rejected(self):
        cached = self.load()
        torch.save(cached, self.cache)
        with self.assertRaisesRegex(ValueError, "legacy"):
            self.load()
        self.cache.unlink()
        self.load()
        payload = torch.load(self.cache, weights_only=True)
        payload["rest_embed"][0, 0, 0] += 1
        torch.save(payload, self.cache)
        with self.assertRaisesRegex(ValueError, "contents changed"):
            self.load()

    def test_probe_validation_requires_persisted_training_cache(self):
        cfg = tiny_tokenizer_cfg()
        cfg.save_path = str(self.checkpoint)
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, encode=lambda text: [2, 3])
        example = SimpleNamespace(prefix_text="prefix", target_text="target")
        spec = SimpleNamespace(generator_n_examples=5, generate_examples=lambda n, seed: [example] * n)
        generator = SimpleNamespace(vqvae=self.vq, pre_tok=tokenizer)
        with self.assertRaisesRegex(FileNotFoundError, "Persisted training cache"):
            validate_cache(generator, spec, cfg, SimpleNamespace(prefix_len=4), Path(self.tmp.name))

    def test_mathematical_validation_rejects_wrong_values_in_full_and_partial_batches(self):
        # Cover a complete middle batch and the final two-row cache batch.
        self.prefix = self.prefix.repeat(26, 1)
        self.target = self.target.repeat(26, 1)
        cached = self.load()
        for field in ("idx", "rest_embed"):
            for row in (100, 129):
                with self.subTest(field=field, row=row):
                    wrong = {name: value.clone() for name, value in cached.items()}
                    wrong[field][row, 0] += 1
                    with self.assertRaises(AssertionError):
                        _validate_cached_values(self.vq, self.target, wrong)


if __name__ == "__main__":
    unittest.main()
