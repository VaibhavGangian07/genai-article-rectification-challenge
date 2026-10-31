"""
End-to-end check of `python rectifier.py rectify-all` with a mocked LLM:
every mapped article runs exactly once, outputs land at the mapped file names
(redirected to a temporary folder), annotations are gone, UTF-8 survives, and
the exit code reports fallbacks and failures. No test reaches a real provider
or writes to rectified_articles/.
"""

import io
import os
import sys
import json
import hashlib
import tempfile
import unittest
import collections
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client
import rectifier
from llm_client import LLMError, LLMResponse
from article_utils import strip_annotations, find_annotation_leak

ARTICLES = json.loads((ROOT / "article_mapping.json").read_text())


def tree_hash(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.glob("*")):
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


class TestRectifyAllCLI(unittest.TestCase):

    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        cwd = os.getcwd()
        os.chdir(ROOT)
        self.addCleanup(os.chdir, cwd)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.out = Path(tmp.name)
        no_network = AssertionError("network call attempted")
        mock.patch.object(llm_client, "_completion", side_effect=no_network).start()
        mock.patch.object(llm_client.requests, "get", side_effect=no_network).start()
        real_mapping = rectifier.get_article_mapping

        def redirected(article_id):
            entry = dict(real_mapping(article_id))
            entry["rectified_file"] = str(self.out / Path(entry["rectified_file"]).name)
            return entry
        mock.patch.object(rectifier, "get_article_mapping", side_effect=redirected).start()
        self.calls = collections.Counter()

    def cli(self, broken):
        """Run rectify-all; article ids in broken get a truncated reply (fallback) or a save error (failed)."""
        def complete(messages, max_tokens, article_id=""):
            self.calls[article_id] += 1
            if broken.get(article_id) == "fallback":
                raise LLMError("truncated", "stopped at max_tokens")
            return LLMResponse(text='{"edits": []}', finish_reason="stop", prompt_tokens=100,
                               completion_tokens=10, reasoning_tokens=5, cost_usd=0.0001)
        mock.patch.object(llm_client, "complete", side_effect=complete).start()
        real_save = rectifier.save_rectified_article

        def save(article_id, content):
            if broken.get(article_id) == "failed":
                raise OSError("disk full")
            real_save(article_id, content)
        mock.patch.object(rectifier, "save_rectified_article", side_effect=save).start()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = rectifier.main(["rectify-all"])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_all_articles_processed_once_and_written_to_their_mapped_files(self):
        before = tree_hash(ROOT / "rectified_articles"), tree_hash(ROOT / "reference_rectified")
        code, stdout, _ = self.cli(broken={})
        ids = [a["article_id"] for a in ARTICLES]
        self.assertEqual(len(set(ids)), 104)
        self.assertEqual(code, 0)
        self.assertEqual(self.calls, collections.Counter(ids))   # each article exactly one request
        expected = sorted(Path(a["rectified_file"]).name for a in ARTICLES)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), expected)
        for article in ARTICLES:
            raw = (self.out / Path(article["rectified_file"]).name).read_bytes()
            text = raw.decode("utf-8")   # strict
            ai = (ROOT / article["ai_generated_file"]).read_text(encoding="utf-8")
            self.assertTrue(text.strip())
            self.assertIsNone(find_annotation_leak(text))
            self.assertNotIn("�", text)
            self.assertEqual(text.replace("\r\n", "\n"), strip_annotations(ai))
        self.assertIn("Completed! Processed 104 articles.", stdout)
        self.assertIn("No change:      104", stdout)
        self.assertEqual(before, (tree_hash(ROOT / "rectified_articles"), tree_hash(ROOT / "reference_rectified")))

    def test_fallback_and_failure_are_reported_and_give_a_nonzero_exit(self):
        code, stdout, stderr = self.cli(broken={"article_004": "fallback", "article_050": "failed"})
        self.assertEqual(code, 1)
        self.assertIn("No change:      102", stdout)
        self.assertIn("Fallback:       1", stdout)
        self.assertIn("Failed:         1", stdout)
        self.assertIn("article_004: truncated", stdout)
        self.assertIn("article_050: OSError: disk full", stdout)
        self.assertIn("[FAIL] article_004", stderr)
        self.assertNotIn("[RECTIFIED] article_004", stdout)
        self.assertEqual(len(list(self.out.iterdir())), 103)   # the failed article has no output
        fallback = (self.out / "article_004.txt").read_text(encoding="utf-8")
        ai = (ROOT / "ai_generated_articles" / "article_004.txt").read_text(encoding="utf-8")
        self.assertEqual(fallback, strip_annotations(ai))       # annotation-stripped, not counted as rectified

    def test_one_failure_does_not_stop_later_articles(self):
        code, _, _ = self.cli(broken={"article_001": "fallback"})
        self.assertEqual(code, 1)
        self.assertEqual(sum(self.calls.values()), 104)
        self.assertEqual(len(list(self.out.iterdir())), 104)


if __name__ == "__main__":
    unittest.main()
