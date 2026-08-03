import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from visualize import AnalysisData, load_titles


class FakeTokenizer:
    def decode(self, token_ids, **_kwargs):
        return f"<{token_ids[0]}>"

    def convert_ids_to_tokens(self, token_id):
        return str(token_id)


class AnalysisDataTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.analysis_path = directory / "analysis.npz"
        self.names_path = directory / "feature_names.jsonl"

        metadata = {
            "format": "csr",
            "model_id": "test/model",
            "dataset_id": "test/data",
            "processed_tokens": 8,
            "context_count": 2,
            "context_size": 4,
            "layer_index": 1,
            "residual_location": "layer_input",
            "d_sae": 4,
            "k": 2,
        }
        np.savez_compressed(
            self.analysis_path,
            metadata=json.dumps(metadata),
            token_ids=np.arange(10, 18, dtype=np.uint32),
            context_ptr=np.array([0, 4, 8], dtype=np.uint32),
            row_ptr=np.array([0, 1, 2, 4, 4, 5, 6, 7, 8], dtype=np.uint32),
            feature_ids=np.array([1, 2, 1, 2, 1, 1, 2, 1], dtype=np.uint32),
            values=np.array([0.2, 0.5, 0.9, 0.3, 0.8, 0.1, 0.7, 0.4], dtype=np.float32),
        )
        self.names_path.write_text(
            '{"feature_id": 1, "title": "Test title"}\n', encoding="utf-8"
        )
        self.data = AnalysisData(
            self.analysis_path, self.names_path, tokenizer=FakeTokenizer()
        )

    def tearDown(self):
        self.temporary.cleanup()

    def test_summary_only_contains_active_features_and_uses_titles(self):
        features = self.data.summary()["features"]

        self.assertEqual([feature["id"] for feature in features], [1, 2])
        self.assertEqual(features[0]["title"], "Test title")
        self.assertEqual(features[0]["activation_count"], 5)
        self.assertAlmostEqual(features[0]["max_activation"], 0.9)

    def test_contexts_are_grouped_and_sorted_by_peak(self):
        result = self.data.feature_contexts(1)

        self.assertEqual(result["activation_count"], 5)
        self.assertEqual(result["context_count"], 2)
        self.assertEqual(
            [context["context_id"] for context in result["contexts"]], [0, 1]
        )
        self.assertAlmostEqual(result["contexts"][0]["peak_activation"], 0.9)
        self.assertEqual(
            result["contexts"][0]["tokens"], ["<10>", "<11>", "<12>", "<13>"]
        )
        np.testing.assert_allclose(
            result["contexts"][0]["activations"], [0.2, 0.0, 0.9, 0.0]
        )

    def test_context_pagination_preserves_peak_order(self):
        result = self.data.feature_contexts(1, offset=1, limit=1)

        self.assertEqual(len(result["contexts"]), 1)
        self.assertEqual(result["contexts"][0]["context_id"], 1)
        self.assertAlmostEqual(result["contexts"][0]["peak_activation"], 0.8)

    def test_missing_feature_is_rejected(self):
        with self.assertRaises(KeyError):
            self.data.feature_contexts(0)


class LoadTitlesTest(unittest.TestCase):
    def test_invalid_jsonl_reports_its_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "names.jsonl"
            path.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 1"):
                load_titles(path)


if __name__ == "__main__":
    unittest.main()
