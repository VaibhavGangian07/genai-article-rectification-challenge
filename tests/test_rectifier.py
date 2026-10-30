import io
import sys
import types
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Stand in for the LLM module so these tests never import litellm or call an API
_stub = types.ModuleType("rectification_system")
def _no_api(_content):
    raise RuntimeError("API calls are disabled in tests")
_stub.run = _no_api
sys.modules["rectification_system"] = _stub

import rectifier
from article_utils import configure_console

AI_TEXT = "## Title\nBody with ₹1,34,900.\n\n**Error Annotations:**\n[\n  {\"error\": \\\"bad\\\"\n"
STRIPPED = "## Title\nBody with ₹1,34,900."


class RectifierTestCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.mapping = {}
        for article_id in ("article_901", "article_902", "article_903"):
            src = self.tmp / "ai" / f"{article_id}.txt"
            src.parent.mkdir(exist_ok=True)
            src.write_text(AI_TEXT, encoding="utf-8")
            self.add(article_id, src, self.tmp / "out" / f"{article_id}.txt")
        patcher = mock.patch.object(rectifier, "get_article_mapping", side_effect=self.lookup)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.stdout, self.stderr = io.StringIO(), io.StringIO()
        for name, stream in (("stdout", self.stdout), ("stderr", self.stderr)):
            p = mock.patch.object(sys, name, stream)
            p.start()
            self.addCleanup(p.stop)

    def add(self, article_id, ai_file, out_file):
        self.mapping[article_id] = {
            "article_id": article_id,
            "ai_generated_file": str(ai_file),
            "rectified_file": str(out_file),
        }

    def lookup(self, article_id):
        if article_id not in self.mapping:
            raise ValueError(f"Article {article_id} not found in mapping")
        return self.mapping[article_id]

    def output(self, article_id):
        return Path(self.mapping[article_id]["rectified_file"]).read_text(encoding="utf-8")


class TestFallback(RectifierTestCase):

    def test_exception_writes_stripped_fallback(self):
        with mock.patch.object(rectifier, "run", side_effect=TimeoutError("api down")):
            content = rectifier.rectify_article("article_901")
        self.assertEqual(content, STRIPPED)
        self.assertEqual(self.output("article_901"), STRIPPED)
        self.assertIn("[FAIL] article_901: TimeoutError: api down", self.stderr.getvalue())
        self.assertIn("[FALLBACK] Rectified article_901", self.stdout.getvalue())

    def test_empty_or_none_output_uses_fallback(self):
        for bad in ("", "   \n", None):
            with self.subTest(output=bad), mock.patch.object(rectifier, "run", return_value=bad):
                self.assertEqual(rectifier.rectify_article("article_901"), STRIPPED)
                self.assertEqual(self.output("article_901"), STRIPPED)

    def test_successful_output_saved_unchanged(self):
        with mock.patch.object(rectifier, "run", return_value="Corrected text.\n"):
            self.assertEqual(rectifier.rectify_article("article_901"), "Corrected text.\n")
        self.assertEqual(self.output("article_901"), "Corrected text.\n")
        self.assertIn("[OK] Rectified article_901", self.stdout.getvalue())
        self.assertEqual(self.stderr.getvalue(), "")

    def test_batch_continues_and_reports_failures(self):
        self.add("article_904", self.tmp / "ai" / "missing.txt", self.tmp / "out" / "article_904.txt")
        articles = [{"article_id": a} for a in ("article_901", "article_904", "article_902")]
        # article_904 fails while reading its input, so run() is only called twice
        with mock.patch.object(rectifier, "run", side_effect=["Fixed one.", RuntimeError("boom")]) as run:
            outcomes = rectifier.process_articles(articles)
        self.assertEqual(run.call_count, 2)
        self.assertEqual(outcomes, {"ok": ["article_901"], "fallback": ["article_902"], "failed": ["article_904"]})
        self.assertEqual(self.output("article_901"), "Fixed one.")
        self.assertEqual(self.output("article_902"), STRIPPED)
        self.assertIn("[FAIL] article_904: FileNotFoundError", self.stderr.getvalue())
        self.assertIn("Failed:    1 article_904", self.stdout.getvalue())

    def test_refuses_to_write_reference_directory(self):
        self.add("article_905", self.mapping["article_901"]["ai_generated_file"],
                 rectifier.REFERENCE_DIR / "article_905.txt")
        with self.assertRaises(ValueError):
            rectifier.save_rectified_article("article_905", "x")
        self.assertFalse((rectifier.REFERENCE_DIR / "article_905.txt").exists())


class TestConsoleOutput(RectifierTestCase):

    def cp1252_stream(self):
        return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

    def test_non_ascii_failure_message_on_cp1252_console(self):
        stdout, stderr = self.cp1252_stream(), self.cp1252_stream()
        with mock.patch.object(sys, "stdout", stdout), mock.patch.object(sys, "stderr", stderr):
            configure_console()
            with mock.patch.object(rectifier, "run", side_effect=ValueError("bad ₹ ✓ value")):
                outcomes = rectifier.process_articles([{"article_id": "article_901"}])
            stderr.flush()
            logged = stderr.buffer.getvalue().decode("cp1252")
        self.assertEqual(outcomes["fallback"], ["article_901"])
        self.assertIn("\\u20b9", logged)

    def test_progress_output_is_ascii(self):
        with mock.patch.object(rectifier, "run", side_effect=["ok", RuntimeError("x")]):
            rectifier.process_articles([{"article_id": "article_901"}, {"article_id": "article_902"}])
        self.assertTrue((self.stdout.getvalue() + self.stderr.getvalue()).isascii())


if __name__ == "__main__":
    unittest.main()
