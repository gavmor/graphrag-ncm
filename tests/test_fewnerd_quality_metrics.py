"""Consumer-visible scoring, serialization, and grounded matching boundaries."""

import unittest

from ncm_graphrag.data.extraction import COMPLETION_DELIMITER, RECORD_DELIMITER
from ncm_graphrag.data.fewnerd_quality_metrics import dictionary_predictions, entities, name_spans, score_rows
from ncm_graphrag.data.real_extraction import _record


def target(*pairs):
    return RECORD_DELIMITER.join([_record(name, kind) for name, kind in pairs] + [COMPLETION_DELIMITER])


class QualityMetricsTests(unittest.TestCase):
    def test_mean_f1_is_not_aggregate_micro_f1(self):
        gold = [target(("Alice", "PERSON")), target(("Paris", "LOCATION"), ("Berlin", "LOCATION"))]
        predicted = [gold[0], target(("Paris", "LOCATION"))]
        scores = score_rows(gold, predicted)
        self.assertAlmostEqual(scores["entity_f1"], 250 / 3)
        self.assertEqual(scores["micro_precision"], 100)
        self.assertAlmostEqual(scores["micro_recall"], 200 / 3)
        self.assertEqual(scores["micro_f1"], 80)
        self.assertEqual(scores["entity_set_exact"], 50)

    def test_registered_validity_does_not_imply_completion(self):
        gold = target(("Alice", "PERSON"))
        scores = score_rows([gold], [gold + "trailing junk"])
        self.assertEqual(scores["validity"], 100)
        self.assertEqual(scores["strict_validity"], 0)
        self.assertEqual(scores["entity_f1"], 100)
        self.assertEqual(scores["target_exact"], 0)
        for malformed in (gold.replace("PERSON", "UNKNOWN"), gold.removesuffix(COMPLETION_DELIMITER)):
            with self.assertRaises(ValueError):
                name_spans(malformed)

    def test_unicode_punctuation_name_offsets(self):
        gold = target(("São-Paulo", "LOCATION"), ("O’Brien", "PERSON"))
        spans = name_spans(gold)
        self.assertEqual(
            [(name, kind) for name, kind, _, _ in spans], [("São-Paulo", "LOCATION"), ("O’Brien", "PERSON")]
        )
        for name, _, start, end in spans:
            self.assertEqual(gold[start:end], name)

    def test_dictionary_longest_overlap_candidate_types_and_boundaries(self):
        training = [
            {"target_text": target(("New York", "LOCATION"), ("York", "LOCATION"))},
            {"target_text": target(("Alice", "PERSON"))},
        ]
        predictions = dictionary_predictions(
            training,
            [
                {"text_unit": "Yorkshire and New York welcome Alice", "entity_types": "LOCATION"},
                {"text_unit": "Malice is not Alice", "entity_types": "PERSON"},
            ],
        )
        self.assertEqual(entities(predictions[0]), {("NEW YORK", "LOCATION")})
        self.assertEqual(entities(predictions[1]), {("ALICE", "PERSON")})

    def test_dictionary_unicode_expansion_and_no_heldout_gold(self):
        training = [{"target_text": target(("STRASSE", "LOCATION"), ("Café", "BUILDING"))}]
        predictions = dictionary_predictions(
            training,
            [
                {"text_unit": "Straße and Cafe\u0301", "entity_types": "LOCATION,BUILDING"},
                {"text_unit": "Atlantis", "entity_types": "LOCATION"},
            ],
        )
        self.assertEqual(entities(predictions[0]), {("STRASSE", "LOCATION"), ("CAFÉ", "BUILDING")})
        self.assertEqual(entities(predictions[1]), set())

    def test_empty_or_unpaired_rows_fail_closed(self):
        for gold, predicted in (([], []), ([target(("Alice", "PERSON"))], [])):
            with self.assertRaises(ValueError):
                score_rows(gold, predicted)


if __name__ == "__main__":
    unittest.main()
