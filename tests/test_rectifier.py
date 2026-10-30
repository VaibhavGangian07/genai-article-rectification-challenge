import io
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rectifier
from article_utils import configure_console, find_annotation_leak

AI_TEXT = "## Title\nBody with ₹1,34,900.\n\n**Error Annotations:**\n[\n  {\"error\": \\\"bad\\\"\n"
STRIPPED = "## Title\nBody with ₹1,34,900."
SOURCE = "Source text."


def result(text, status="rectified", reason=""):
    return SimpleNamespace(text=text, status=status, reason=reason)


class RectifierTestCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.mapping = {}
        for article_id in ("article_901", "article_902", "article_903"):
            self.add(article_id, self.write("ai", article_id, AI_TEXT), self.tmp / "out" / f"{article_id}.txt",
                     self.write("src", article_id, SOURCE))
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(rectifier, "get_article_mapping", side_effect=self.lookup).start()
        # No test may reach the real engine (and so the API) by accident
        self.run_mock = mock.patch.object(rectifier, "run", side_effect=AssertionError("API calls are disabled in tests")).start()
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        mock.patch.object(sys, "stdout", self.stdout).start()
        mock.patch.object(sys, "stderr", self.stderr).start()

    def write(self, folder, article_id, text):
        path = self.tmp / folder / f"{article_id}.txt"
        path.parent.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def add(self, article_id, ai_file, out_file, source_file):
        self.mapping[article_id] = {
            "article_id": article_id,
            "source_file": str(source_file),
            "ai_generated_file": str(ai_file),
            "rectified_file": str(out_file),
        }

    def lookup(self, article_id):
        if article_id not in self.mapping:
            raise ValueError(f"Article {article_id} not found in mapping")
        return self.mapping[article_id]

    def output(self, article_id):
        return Path(self.mapping[article_id]["rectified_file"]).read_text(encoding="utf-8")


class TestRectify(RectifierTestCase):

    def test_engine_receives_ai_article_and_source(self):
        self.run_mock.side_effect = None
        self.run_mock.return_value = result("Corrected text.")
        rectifier.rectify_article("article_901")
        self.run_mock.assert_called_once_with(AI_TEXT, SOURCE, "article_901")

    def test_successful_output_saved_unchanged(self):
        self.run_mock.side_effect = None
        self.run_mock.return_value = result("Corrected text.\n")
        self.assertEqual(rectifier.rectify_article("article_901"), "Corrected text.\n")
        self.assertEqual(self.output("article_901"), "Corrected text.\n")
        self.assertIn("[RECTIFIED] article_901", self.stdout.getvalue())
        self.assertEqual(self.stderr.getvalue(), "")

    def test_engine_fallback_is_logged_with_its_reason(self):
        self.run_mock.side_effect = None
        self.run_mock.return_value = result(STRIPPED, "fallback", "budget_exceeded")
        rectifier.rectify_article("article_901")
        self.assertEqual(self.output("article_901"), STRIPPED)
        self.assertIn("[FAIL] article_901: budget_exceeded", self.stderr.getvalue())
        self.assertIn("[FALLBACK] article_901", self.stdout.getvalue())


class TestFallback(RectifierTestCase):

    def test_exception_writes_stripped_fallback(self):
        self.run_mock.side_effect = TimeoutError("api down")
        content = rectifier.rectify_article("article_901")
        self.assertEqual(content, STRIPPED)
        self.assertEqual(self.output("article_901"), STRIPPED)
        self.assertIn("[FAIL] article_901: TimeoutError: api down", self.stderr.getvalue())
        self.assertIn("[FALLBACK] article_901", self.stdout.getvalue())

    def test_empty_or_none_output_uses_fallback(self):
        self.run_mock.side_effect = None
        for bad in ("", "   \n", None):
            with self.subTest(output=bad):
                self.run_mock.return_value = result(bad)
                self.assertEqual(rectifier.rectify_article("article_901"), STRIPPED)
                self.assertEqual(self.output("article_901"), STRIPPED)

    def test_output_with_annotation_text_is_never_saved(self):
        self.run_mock.side_effect = None
        for leaked in ("Text.\n\n**Error Annotations:**\n[]", 'Text "correction": "x"', "Text ```"):
            with self.subTest(output=leaked):
                self.run_mock.return_value = result(leaked)
                rectifier.rectify_article("article_901")
                self.assertEqual(self.output("article_901"), STRIPPED)

    def test_unreadable_source_uses_fallback(self):
        self.mapping["article_901"]["source_file"] = str(self.tmp / "src" / "missing.txt")
        rectifier.rectify_article("article_901")
        self.assertEqual(self.output("article_901"), STRIPPED)
        self.assertIn("FileNotFoundError", self.stderr.getvalue())
        self.run_mock.assert_not_called()

    def test_batch_reports_each_status(self):
        self.add("article_904", self.tmp / "ai" / "missing.txt", self.tmp / "out" / "article_904.txt",
                 self.tmp / "src" / "article_901.txt")
        articles = [{"article_id": a} for a in ("article_901", "article_904", "article_902", "article_903")]
        # article_904 fails while reading its input, so run() is only called three times
        self.run_mock.side_effect = [result("Fixed one."), RuntimeError("boom"), result(STRIPPED, "no_change")]
        outcomes = rectifier.process_articles(articles)
        self.assertEqual(self.run_mock.call_count, 3)
        self.assertEqual(outcomes, {"rectified": ["article_901"], "no_change": ["article_903"], "no_valid_edits": [],
                                    "fallback": ["article_902"], "failed": ["article_904"]})
        self.assertEqual(self.output("article_902"), STRIPPED)
        summary = self.stdout.getvalue()
        self.assertIn("Rectified:      1", summary)
        self.assertIn("Fallback:       1 (annotation-stripped text, not corrected)", summary)
        self.assertIn("article_902: RuntimeError: boom", summary)
        self.assertIn("article_904: FileNotFoundError", summary)


class TestSaveGuards(RectifierTestCase):

    def test_refuses_to_write_reference_directory(self):
        self.add("article_905", self.mapping["article_901"]["ai_generated_file"],
                 rectifier.REFERENCE_DIR / "article_905.txt", SOURCE)
        with self.assertRaises(ValueError):
            rectifier.save_rectified_article("article_905", "x")
        self.assertFalse((rectifier.REFERENCE_DIR / "article_905.txt").exists())

    def test_refuses_to_save_annotation_text(self):
        with self.assertRaises(ValueError):
            rectifier.save_rectified_article("article_901", 'Body\n\n**Error Annotations:**\n[{"error": "x"}]')
        self.assertFalse(Path(self.mapping["article_901"]["rectified_file"]).exists())

    def test_all_104_fallback_outputs_pass_the_leak_guard(self):
        real = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}
        for article_id, entry in real.items():
            self.add(article_id, ROOT / entry["ai_generated_file"], self.tmp / "out104" / f"{article_id}.txt",
                     ROOT / entry["source_file"])
        self.run_mock.side_effect = RuntimeError("no engine")
        outcomes = rectifier.process_articles([{"article_id": a} for a in real])
        self.assertEqual(len(outcomes["fallback"]), 104)
        for article_id, entry in real.items():
            with self.subTest(article=article_id):
                out = self.output(article_id)
                original = (ROOT / entry["ai_generated_file"]).read_text(encoding="utf-8")
                self.assertIsNone(find_annotation_leak(out))
                self.assertTrue(out.strip() and original.startswith(out))


class TestConsoleOutput(RectifierTestCase):

    def cp1252_stream(self):
        return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    def test_non_ascii_failure_message_on_cp1252_console(self):
        stdout, stderr = self.cp1252_stream(), self.cp1252_stream()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            configure_console()
            self.run_mock.side_effect = ValueError("bad ₹ ✓ value")
            outcomes = rectifier.process_articles([{"article_id": "article_901"}])
            stderr.flush()
            logged = stderr.buffer.getvalue().decode("cp1252")
        self.assertEqual(outcomes["fallback"], ["article_901"])
        self.assertIn("\\u20b9", logged)

    def test_progress_output_is_ascii(self):
        self.run_mock.side_effect = [result("ok"), RuntimeError("x")]
        rectifier.process_articles([{"article_id": "article_901"}, {"article_id": "article_902"}])
        self.assertTrue((self.stdout.getvalue() + self.stderr.getvalue()).isascii())


if __name__ == "__main__":
    unittest.main()
