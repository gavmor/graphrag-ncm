"""Consumer-visible checks for corpus relocation and bounded graph targets."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ncm_graphrag.data import corpora
from ncm_graphrag.data import rebel_extraction as rebel
from ncm_graphrag.data.extraction import COMPLETION_DELIMITER, parse_records


class CharacterTokenizer:
    def encode(self, text):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


def triple(subject, predicate, object_):
    return {
        "subject": {"surfaceform": subject},
        "predicate": {"surfaceform": predicate},
        "object": {"surfaceform": object_},
    }


class DatasetTests(unittest.TestCase):
    def test_external_corpus_root_resolves_old_and_new_manifest_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "information_extraction" / "rebel").mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps({"rebel": {"path": "corpora/information_extraction/rebel"}}), encoding="utf-8"
            )
            with patch.dict("os.environ", {"NCM_CORPORA_ROOT": tmp}):
                self.assertEqual(corpora.get_corpus_path("rebel"), root / "information_extraction" / "rebel")
                (root / "manifest.json").write_text(
                    json.dumps({"rebel": {"path": "information_extraction/rebel"}}), encoding="utf-8"
                )
                self.assertEqual(corpora.get_corpus_path("rebel"), root / "information_extraction" / "rebel")

    def test_exact_surface_names_direction_and_predicate(self):
        record = {
            "docid": "1",
            "text": "Alice founded Acme Labs in Paris.",
            "triples": [triple("Alice", "founder of", "Acme Labs")],
        }
        with patch.object(rebel, "_pre_tok", return_value=CharacterTokenizer()):
            with patch.object(rebel, "TARGET_TOKEN_BUDGET", 512):
                example = rebel.example_from_rebel(record)
        assert example is not None
        parsed = parse_records(example.target_text)
        self.assertEqual({e["name"] for e in parsed["entities"]}, {"ALICE", "ACME LABS"})
        self.assertIn("<|>Alice<|>Acme Labs<|>founder of<|>1", example.target_text)
        self.assertEqual(parsed["relationships"][0]["description"], "founder of")
        self.assertTrue(example.target_text.endswith(COMPLETION_DELIMITER))

    def test_never_emit_dangling_or_out_of_source_relations(self):
        record = {
            "text": "Alice founded Acme Labs in Paris.",
            "triples": [triple("Alice", "founder of", "Acme Labs"), triple("Alice", "country", "Atlantis")],
        }
        with patch.object(rebel, "_pre_tok", return_value=CharacterTokenizer()):
            with patch.object(rebel, "TARGET_TOKEN_BUDGET", 512):
                example = rebel.example_from_rebel(record)
        assert example is not None
        parsed = parse_records(example.target_text)
        self.assertEqual(len(parsed["relationships"]), 1)
        self.assertNotIn("Atlantis", example.target_text)
        self.assertIsNone(rebel.example_from_rebel({"text": "Alice", "triples": []}))

    def test_budget_rejects_whole_triple_instead_of_truncating_tuple(self):
        record = {
            "text": "Alice founded Acme Labs in Paris.",
            "triples": [triple("Alice", "founder of", "Acme Labs"), triple("Acme Labs", "city", "Paris")],
        }
        with patch.object(rebel, "_pre_tok", return_value=CharacterTokenizer()):
            with patch.object(rebel, "TARGET_TOKEN_BUDGET", 240):
                example = rebel.example_from_rebel(record)
        assert example is not None
        self.assertLessEqual(len(example.target_text) + 1, 240)
        parsed = parse_records(example.target_text)
        self.assertEqual(len(parsed["relationships"]), 1)
        self.assertTrue(rebel.is_well_formed(example.target_text))

    def test_metric_distinguishes_relation_direction_and_label(self):
        gold = (
            '("entity"<|>Alice<|>UNKNOWN<|>x)##("entity"<|>Bob<|>UNKNOWN<|>x)##'
            '("relationship"<|>Alice<|>Bob<|>parent<|>1)##<|COMPLETE|>'
        )
        reverse = gold.replace("Alice<|>Bob<|>parent", "Bob<|>Alice<|>parent")
        wrong_label = gold.replace("parent", "sibling")
        self.assertEqual(rebel.graph_sets(gold)[0], rebel.graph_sets(reverse)[0])
        self.assertFalse(rebel.graph_sets(gold)[2] & rebel.graph_sets(reverse)[2])
        self.assertFalse(rebel.graph_sets(gold)[2] & rebel.graph_sets(wrong_label)[2])

    def test_validation_selection_and_reporting_do_not_share_article(self):
        examples = [
            rebel.GraphExtractionExample("source", "target", f"article-{index // 2}", "rebel") for index in range(400)
        ]
        with patch.object(rebel, "_split_examples", return_value=examples):
            selected, heldout = rebel.validation_partitions(30)
        self.assertEqual(len(selected), 30)
        self.assertEqual(len(heldout), 30)
        self.assertFalse({e.source_id for e in selected} & {e.source_id for e in heldout})

    def test_gold_scoring_does_not_drop_relations_to_fit_training_budget(self):
        triples = [
            rebel._Triple("Alice", "Acme", "founder"),
            rebel._Triple("Acme", "Paris", "location"),
        ]
        with patch.object(rebel, "_pre_tok", return_value=CharacterTokenizer()):
            with patch.object(rebel, "TARGET_TOKEN_BUDGET", 200):
                gold = rebel._bounded_example(
                    "Alice Acme Paris", triples, source_id="gold", dataset="docred", cap_target=False
                )
        assert gold is not None
        self.assertEqual(len(parse_records(gold.target_text)["relationships"]), 2)
        self.assertGreater(len(gold.target_text), 200)


if __name__ == "__main__":
    unittest.main()
