import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from article_utils import (split_annotations, strip_annotations, parse_annotations, find_annotation_leak,
                           normalize_with_map)

BODY = "## Title\n### Synopsis\nFirst paragraph.\n\n## Section\nSecond paragraph."


class TestAnnotationRemoval(unittest.TestCase):

    def assertStripped(self, text, annotations_expected=True):
        body, annotations = split_annotations(text)
        self.assertEqual(body, BODY)
        if annotations_expected:
            self.assertIsNotNone(annotations)
        else:
            self.assertIsNone(annotations)

    def test_bold_header(self):
        self.assertStripped(BODY + '\n\n**Error Annotations:**\n[\n    {"error": "x"}\n]')

    def test_markdown_header(self):
        self.assertStripped(BODY + '\n\n## Error Annotations:\n[\n    {"error": "x"}\n]')

    def test_header_variants(self):
        for header in ("**Error Annotations**:", "### Error Annotations", "Error Annotations:",
                       "**error annotations:**", "  **Error Annotations:**  ",
                       '**Error Annotations:** [{"error": "x"}]'):
            with self.subTest(header=header):
                self.assertStripped(BODY + "\n\n" + header + "\n[]")

    def test_malformed_json(self):
        # Shape of article_015 / article_047: escaped quote outside a string
        self.assertStripped(BODY + '\n\n**Error Annotations:**\n[\n    {\n        "error": \\"broken\\",\n')

    def test_empty_list_with_stray_fence(self):
        # Shape of article_053
        self.assertStripped(BODY + "\n\n**Error Annotations:**\n[]\n```")

    def test_header_with_nothing_after(self):
        self.assertStripped(BODY + "\n\n**Error Annotations:**")

    def test_missing_annotations(self):
        self.assertStripped(BODY, annotations_expected=False)
        self.assertStripped(BODY + "\n", annotations_expected=False)

    def test_inline_mention_is_not_a_header(self):
        text = "The report listed Error Annotations: none were found.\n\nMore text."
        self.assertEqual(split_annotations(text), (text, None))

    def test_crlf_line_endings(self):
        text = BODY.replace("\n", "\r\n") + "\r\n\r\n**Error Annotations:**\r\n[]\r\n"
        self.assertEqual(strip_annotations(text), BODY.replace("\n", "\r\n"))

    def test_only_first_header_splits(self):
        text = BODY + "\n\n**Error Annotations:**\n[]\n\n## Error Annotations:\n[]"
        body, annotations = split_annotations(text)
        self.assertEqual(body, BODY)
        self.assertIn("## Error Annotations:", annotations)

    def test_idempotent(self):
        text = BODY + "\n\n**Error Annotations:**\n[]"
        self.assertEqual(strip_annotations(strip_annotations(text)), strip_annotations(text))


class TestAnnotationRemovalOnDataset(unittest.TestCase):
    """Runs the stripper over all 104 real AI-generated articles."""

    UNANNOTATED = {"article_001", "article_002", "article_003"}
    # article_033 and article_100 use the "## Error Annotations:" header style

    @classmethod
    def setUpClass(cls):
        cls.articles = {
            p.stem: p.read_text(encoding="utf-8")
            for p in sorted((ROOT / "ai_generated_articles").glob("article_*.txt"))
        }

    def test_all_articles_present(self):
        self.assertEqual(len(self.articles), 104)

    def test_body_is_preserved_prefix(self):
        for article_id, text in self.articles.items():
            with self.subTest(article=article_id):
                body = strip_annotations(text)
                self.assertTrue(body.strip())
                self.assertTrue(text.startswith(body))
                self.assertEqual(body, body.rstrip())

    def test_no_annotation_artifacts_remain(self):
        for article_id, text in self.articles.items():
            with self.subTest(article=article_id):
                body = strip_annotations(text)
                self.assertNotIn("error annotations", body.lower())
                self.assertNotIn("```", body)
                self.assertNotIn('"location"', body)
                self.assertNotIn('"error_type"', body)

    def test_annotation_presence(self):
        unannotated = {a for a, t in self.articles.items() if split_annotations(t)[1] is None}
        self.assertEqual(unannotated, self.UNANNOTATED)

    def test_known_edge_cases(self):
        for article_id in ("article_015", "article_033", "article_047", "article_053", "article_100"):
            with self.subTest(article=article_id):
                body, annotations = split_annotations(self.articles[article_id])
                self.assertIsNotNone(annotations)
                self.assertTrue(annotations.strip().startswith("["))


class TestAnnotationParsing(unittest.TestCase):

    def test_valid_json(self):
        raw = '\n[\n {"location": "p1", "error": "\\"on Monday\\"", "correction": "\\"on Tuesday\\"", "error_type": "t"}\n]'
        self.assertEqual(parse_annotations(raw), [{"error": '"on Monday"', "correction": '"on Tuesday"'}])

    def test_malformed_json_falls_back_to_fields(self):
        raw = ('\n[\n    {\n        "location": "p1",\n        "error": \\"broken quote\\",\n'
               '        "correction": "fixed value",\n    },\n    {\n        "error": "second",\n'
               '        "correction": "two"\n    }\n')
        self.assertEqual(parse_annotations(raw), [{"error": "broken quote", "correction": "fixed value"},
                                                  {"error": "second", "correction": "two"}])

    def test_empty_or_missing(self):
        for raw in (None, "", "\n[]\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(parse_annotations(raw), [])

    def test_every_annotated_article_yields_hints(self):
        for path in sorted((ROOT / "ai_generated_articles").glob("article_*.txt")):
            raw = split_annotations(path.read_text(encoding="utf-8"))[1]
            if raw is not None and raw.strip().strip("`").strip() != "[]":
                with self.subTest(article=path.stem):
                    entries = parse_annotations(raw)
                    self.assertTrue(entries)
                    self.assertTrue(all(e["error"] and e["correction"] for e in entries))


class TestLeakDetection(unittest.TestCase):

    def test_detects_annotation_artifacts(self):
        for text in ("Body\n\n**Error Annotations:**\n[]", "Body\n## Error Annotations:", 'x "correction": "y"',
                     'x "location": "p1"', "x ```json"):
            with self.subTest(text=text):
                self.assertIsNotNone(find_annotation_leak(text))

    def test_clean_articles_pass(self):
        for path in sorted((ROOT / "reference_rectified").glob("article_*.txt")):
            with self.subTest(article=path.stem):
                self.assertIsNone(find_annotation_leak(path.read_text(encoding="utf-8")))


class TestNormalizeWithMap(unittest.TestCase):

    def test_maps_back_to_original_positions(self):
        text = "said “no” —  then left"
        norm, index_map = normalize_with_map(text)
        self.assertEqual(norm, 'said "no" - then left')
        self.assertEqual(len(norm), len(index_map))
        for i, ch in enumerate(norm):
            if ch not in "\"- ":
                self.assertEqual(text[index_map[i]], ch)


if __name__ == "__main__":
    unittest.main()
