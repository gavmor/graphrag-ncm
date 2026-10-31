"""Behavioral regression checks for optimized coarse-to-fine sampling."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from ncm_graphrag.ncm.generate import NCMTaskGenerator, NextConceptTransformer


class RecordingEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.indices = None

    def forward(self, indices):  # ty: ignore[invalid-mutable-override]
        self.indices = indices.clone()
        return indices.unsqueeze(-1).float()


class PrefixBlock(torch.nn.Module):
    def forward(self, x, attn_mask, rope_cos_sin):  # ty: ignore[invalid-mutable-override]
        return torch.cat((x[:, :1], x[:, 1:] + x[:, :1]), dim=1)


class Head(torch.nn.Module):
    def forward(self, x):  # ty: ignore[invalid-mutable-override]
        # The first coordinate is the prefix: conditional and null logits differ.
        value = x[:, :1, 0]
        return torch.stack((value, torch.zeros_like(value), -value), dim=-1)


class FakeNet(torch.nn.Module):
    _sample_level_raw = NextConceptTransformer._sample_level_raw

    def __init__(self):
        super().__init__()
        self.prefix_len = 1
        self.prefix_pad_id = 0
        self.levels = [1, 1]
        self._lvl_first = 0
        self.use_rope = True
        self.config = SimpleNamespace(contextual_prefix=False)
        self.transformer = SimpleNamespace(
            blocks=torch.nn.ModuleList([PrefixBlock()]),
            ln_f=torch.nn.Identity(),
            wte=torch.nn.Identity(),
            wle=lambda indices: torch.zeros(*indices.shape, 1),
        )
        self.lvl_1L = torch.zeros(1, 2, dtype=torch.long)
        self.lm_heads = torch.nn.ModuleList([Head(), Head()])
        self.rope_cos_gen_0 = self.rope_sin_gen_0 = torch.ones(1)
        self.rope_cos_gen_1 = self.rope_sin_gen_1 = torch.ones(1)

    def build_l0_query_input(self, batch, dtype):
        return torch.ones(batch, 1, 1, dtype=dtype)


class PrefixEmbedding(torch.nn.Module):
    def forward(self, tokens):  # ty: ignore[invalid-mutable-override]
        return tokens.unsqueeze(-1).float()


class SamplingTests(unittest.TestCase):
    def make_generator(self):
        generator = NCMTaskGenerator.__new__(NCMTaskGenerator)
        generator.net = FakeNet()  # ty: ignore[invalid-assignment]
        embedding = RecordingEmbedding()
        vq = SimpleNamespace(
            embeddings=[embedding],
            get_next_autoregressive_input=lambda level, z_hat, z: (z_hat, z),
        )
        generator.vqvae = SimpleNamespace(  # ty: ignore[invalid-assignment]
            vq=vq, embed_dim=1, encoder=SimpleNamespace(wte=PrefixEmbedding())
        )
        generator.device = "cpu"
        generator.compile_blocks = False
        generator._optimize_net()
        return generator, embedding

    def test_sampling_parameters_and_guided_codes(self):
        for guided in (False, True):
            with self.subTest(guided=guided):
                generator, embedding = self.make_generator()
                calls = []

                def sample(logits, temperature, top_k, top_p):
                    calls.append((logits.clone(), temperature, top_k, top_p))
                    return torch.full(logits.shape[:-1], 2, dtype=torch.long)

                generator.net._sample_level_raw = sample
                result = generator.net.generate(
                    None,
                    torch.tensor([[3], [4]]),
                    temperature=0.7,
                    top_k=2,
                    top_p=0.8,
                    cfg_scale=2 if guided else 0,
                    cfg_null_eos=1,
                )
                self.assertEqual(result.tolist(), [[2, 2], [2, 2]])
                self.assertEqual(len(calls), 2)
                for logits, temperature, top_k, top_p in calls:
                    self.assertEqual(logits.shape, (2, 1, 3))
                    self.assertEqual((temperature, top_k, top_p), (0.7, 2, 0.8))
                self.assertEqual(embedding.indices.tolist(), [[2], [2]] * (2 if guided else 1))
                expected = [9.0, 12.0] if guided else [5.0, 6.0]
                torch.testing.assert_close(calls[1][0][:, 0, 0], torch.tensor(expected))

    def test_real_sampler_honors_greedy_and_sampling_filters(self):
        generator, _ = self.make_generator()
        prefix = torch.tensor([[3]] * 64)
        greedy = generator.net.generate(None, prefix, temperature=0, top_k=3, top_p=1)
        self.assertTrue(torch.all(greedy == 0))
        with patch(
            "torch.multinomial",
            side_effect=lambda probs, num_samples: torch.full((probs.shape[0], 1), 1, dtype=torch.long),
        ) as multinomial:
            sampled = generator.net.generate(None, prefix, temperature=0.7, top_k=2, top_p=1)
        self.assertTrue(torch.all(sampled == 1))
        self.assertEqual(multinomial.call_count, 2)
        self.assertTrue(torch.all(multinomial.call_args_list[0].args[0][:, 2] == 0))
        self.assertTrue(torch.all(multinomial.call_args_list[0].args[0][:, 1] > 0))


if __name__ == "__main__":
    unittest.main()
