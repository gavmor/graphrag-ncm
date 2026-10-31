"""Consumer-visible diagnostic boundaries; no downloads or tokenizer cache."""

import importlib
import unittest
from collections import Counter

import torch
from models.vqvae import VQVAE
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from ncm_graphrag.ncm.config import build_tokenizer_cfg
from ncm_graphrag.ncm.fewnerd_quality import alignment, reconstruct, subgroup_scores

GPT2TokenizerFast = getattr(importlib.import_module("transformers"), "GPT2TokenizerFast")


class TokenizerQualityTests(unittest.TestCase):
    def test_subgroups_penalize_prediction_only_rows_and_ignore_type_for_seen(self):
        gold = [{("ALICE", "PERSON")}, {("BOBBY", "PERSON")}]
        predicted = [{("ALICE", "LOCATION")}, {("ALICE", "PERSON")}]
        seen = subgroup_scores(gold, predicted, {"ALICE"}, True)
        self.assertEqual(seen["rows"], 2)
        self.assertEqual(seen["gold_entities"], 1)
        self.assertEqual(seen["predicted_entities"], 2)
        self.assertEqual(seen["typed_true_positives"], 0)
        self.assertEqual(seen["entity_f1"], 0)
        self.assertEqual(seen["name_f1"], 50)
        self.assertAlmostEqual(seen["name_micro_f1"], 200 / 3)
        self.assertEqual(seen["typed_micro_f1_denominator"], 3)
        unseen = subgroup_scores(gold, predicted, {"ALICE"}, False)
        self.assertEqual(unseen["rows"], 1)
        self.assertEqual(unseen["gold_entities"], 1)
        self.assertEqual(unseen["predicted_entities"], 0)
        self.assertEqual(unseen["entity_f1"], 0)

    def test_absent_subgroup_reports_counts_and_none_not_perfect_score(self):
        result = subgroup_scores([{("ALICE", "PERSON")}], [{("ALICE", "PERSON")}], {"ALICE"}, False)
        self.assertEqual(result["rows"], 0)
        self.assertEqual(result["gold_entities"], 0)
        self.assertEqual(result["predicted_entities"], 0)
        self.assertEqual(result["typed_micro_f1_denominator"], 0)
        self.assertEqual(result["name_micro_f1_denominator"], 0)
        for key in (
            "entity_f1",
            "name_f1",
            "typed_micro_f1",
            "name_micro_f1",
            "name_reconstruction_recall",
            "name_reconstruction_precision",
        ):
            self.assertIsNone(result[key])

    def test_unicode_punctuation_alignment_uses_original_character_offsets(self):
        # Real byte-level BPE built in memory: multiple byte tokens can share
        # one source character offset, without a tokenizer cache or downloads.
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        tokenizer = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tok = GPT2TokenizerFast(tokenizer_object=tokenizer, eos_token="<|endoftext|>")
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        name = "São–Paulo’s"
        text = f'("entity"<|>{name}<|>LOCATION<|>named place)##<|COMPLETE|>'
        encoded = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        row = {"target_text": text, "target_ids": encoded["input_ids"] + [tok.eos_token_id]}
        names, types = alignment(row, tok)
        self.assertEqual([(n, kind) for n, kind, _ in names], [(name, "LOCATION")])
        start, end = text.index(name), text.index(name) + len(name)
        positions = names[0][2]
        self.assertTrue(
            all(encoded["offset_mapping"][i][0] < end and encoded["offset_mapping"][i][1] > start for i in positions)
        )
        self.assertTrue(
            all(
                any(encoded["offset_mapping"][i][0] <= c < encoded["offset_mapping"][i][1] for i in positions)
                for c in range(start, end)
            )
        )
        self.assertEqual(len(positions), len(name.encode("utf-8")))
        self.assertTrue(set(positions).isdisjoint(types))
        broken = {**row, "target_ids": row["target_ids"][:-2] + [tok.eos_token_id]}
        with self.assertRaisesRegex(ValueError, "Frozen target IDs"):
            alignment(broken, tok)

    def test_real_reconstruction_preserves_model_buffers_rng_and_requires_eval(self):
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        backend = Tokenizer(models.BPE(vocab={token: i for i, token in enumerate(alphabet)}, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tok = GPT2TokenizerFast(tokenizer_object=backend, eos_token="<|endoftext|>")
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        text = '("entity"<|>Paris<|>LOCATION<|>city)##<|COMPLETE|>'
        row = {"group_id": "one", "target_text": text, "target_ids": tok.encode(text) + [tok.eos_token_id]}
        cfg = build_tokenizer_cfg().vqvae
        cfg.in_vocab_size = len(tok)
        cfg.embed_dim = 16
        cfg.context_length = 80
        cfg.vqvae_vocab_size = [8, 8, 8]
        cfg.vector_quantizer_config.levels.manual_levels = [2, 4, 80]
        with torch.random.fork_rng():
            torch.manual_seed(31)
            model = VQVAE(**cfg).eval()
            before = {name: value.clone() for name, value in model.state_dict().items()}
            rng = torch.get_rng_state().clone()
            _, details = reconstruct(model, [row], tok, {"PARIS"}, Counter(row["target_ids"][:-1]))
            self.assertEqual(details["counts"]["names"], 1)
            self.assertEqual(details["type_exact"]["fields"], 1)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            model.train()
            with self.assertRaisesRegex(ValueError, "model.eval"):
                reconstruct(model, [row], tok, {"PARIS"}, Counter())


if __name__ == "__main__":
    unittest.main()
