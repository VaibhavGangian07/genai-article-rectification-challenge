import io
import sys
import json
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate_baseline

REFERENCE_DIR = ROOT / "reference_rectified"
EXPECTED = [f"article_{i:03d}.txt" for i in range(1, 11)]


class TestReferenceBackup(unittest.TestCase):

    def test_ten_references_present(self):
        self.assertEqual(sorted(p.name for p in REFERENCE_DIR.glob("article_*.txt")), EXPECTED)

    def test_references_match_checksums(self):
        evaluate_baseline.verify_references()

    def test_manifest_covers_every_reference(self):
        names = [line.split()[1] for line in (REFERENCE_DIR / "SHA256SUMS.txt").read_text().splitlines()]
        self.assertEqual(names, EXPECTED)

    def test_mapping_never_targets_reference_directory(self):
        mapping = json.loads((ROOT / "article_mapping.json").read_text())
        for entry in mapping:
            for key in ("source_file", "ai_generated_file", "rectified_file"):
                with self.subTest(article=entry["article_id"], key=key):
                    self.assertFalse((ROOT / entry[key]).resolve().is_relative_to(REFERENCE_DIR))


class TestBaselineEvaluation(unittest.TestCase):

    def test_compare_reports_changed_spans(self):
        result = evaluate_baseline.compare("covers 63 million km", "covers 165 million km")
        self.assertEqual(result["changes"], [{"op": "replace", "candidate": "63", "reference": "165"}])
        self.assertAlmostEqual(result["similarity"], 0.75)

    def test_identical_text_scores_one(self):
        result = evaluate_baseline.compare("same text", "same text")
        self.assertEqual((result["similarity"], result["changes"]), (1.0, []))

    def test_evaluate_all_references(self):
        results = evaluate_baseline.evaluate()
        self.assertEqual([r["article_id"] for r in results], [n[:-4] for n in EXPECTED])
        for r in results:
            self.assertTrue(0.0 <= r["similarity"] <= 1.0)

    def test_report_is_labelled_baseline(self):
        out = io.StringIO()
        with mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "argv", ["evaluate_baseline.py", "--summary"]):
            evaluate_baseline.main()
        self.assertIn("BASELINE", out.getvalue())
        self.assertIn("NOT a factual-accuracy score", out.getvalue())

    def test_detects_modified_reference(self):
        real_read_bytes = Path.read_bytes
        def tampered(path):
            return b"tampered" if path.name == "article_001.txt" else real_read_bytes(path)
        with mock.patch.object(Path, "read_bytes", tampered):
            with self.assertRaises(RuntimeError):
                evaluate_baseline.verify_references()


if __name__ == "__main__":
    unittest.main()
