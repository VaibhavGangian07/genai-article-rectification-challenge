"""
The 8192-token output limit: it reaches the provider, the budget guard
reserves for it, and a reply cut off at the limit is never applied.
No test reaches a real provider.
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import llm_client
import rectification_system
from prompts import build_messages
from article_utils import split_annotations, strip_annotations, find_annotation_leak

ENV = {"LLM_MODEL_NAME": "groq/openai/gpt-oss-120b", "LLM_API_KEY": "dummy-test-key",
       "LLM_API_BASE": "https://api.groq.com/openai/v1"}   # direct-Groq mode; no request ever leaves the test
PRICE = llm_client.PRICING[ENV["LLM_MODEL_NAME"]]
MAPPING = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}
AI, SOURCE = ((ROOT / MAPPING["article_010"][k]).read_text(encoding="utf-8")
              for k in ("ai_generated_file", "source_file"))
EVIDENCE = ("Vedan, who has been granted anticipatory bail in the case by the Kerala High Court, "
            "turned up at the police station by 10 a.m. on Tuesday (September 9, 2025).")
VALID_EDITS = json.dumps({"edits": [{"op": "replace", "find": "on Monday morning", "replace": "on Tuesday morning",
                                     "evidence": EVIDENCE, "evidence_from": "source"}]})


def raw_response(content, finish="stop", completion=500):
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)
    usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=completion,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=300))
    return SimpleNamespace(choices=[choice], usage=usage, _hidden_params={})


class OutputLimitCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        self.ledger = Path(tmp.name) / "ledger.jsonl"
        mock.patch.object(llm_client, "LEDGER_PATH", self.ledger).start()
        mock.patch.object(llm_client, "CACHE_DIR", Path(tmp.name) / "cache").start()
        mock.patch.object(llm_client, "_disabled_reason", None).start()
        mock.patch.object(llm_client, "_sleep").start()
        mock.patch.dict(os.environ, ENV).start()
        os.environ.pop("MAX_SPEND_USD", None)
        self.completion = mock.patch.object(llm_client, "_completion",
                                            side_effect=AssertionError("unexpected API call")).start()
        mock.patch.object(llm_client.requests, "get", side_effect=AssertionError("unexpected HTTP request")).start()

    def run_article(self, *responses):
        self.completion.side_effect = list(responses)
        return rectification_system.run(AI, SOURCE, "article_010")

    def assertSafeFallback(self, result, reason):
        self.assertEqual((result.status, result.reason), ("fallback", reason))
        self.assertEqual(result.text, strip_annotations(AI))
        self.assertEqual(result.applied, [])
        self.assertIsNone(find_annotation_leak(result.text))


class TestOutputLimit(OutputLimitCase):

    def test_request_is_sent_with_8192_output_tokens(self):
        self.assertEqual(rectification_system.MAX_OUTPUT_TOKENS, 8192)
        result = self.run_article(raw_response(VALID_EDITS))
        self.assertEqual(result.status, "rectified")
        self.assertEqual(self.completion.call_args.kwargs["max_tokens"], 8192)

    def test_budget_guard_reserves_the_full_8192_tokens(self):
        messages = build_messages(SOURCE, split_annotations(AI)[0], split_annotations(AI)[1])
        worst = llm_client._worst_case(PRICE, messages, 8192)
        self.assertAlmostEqual(worst - llm_client._worst_case(PRICE, messages, 4096), 4096 * PRICE["output_per_m"] / 1e6)
        # room for a 4096-token reply but not an 8192-token one: the call must not be sent
        self.ledger.write_text(json.dumps({"cost_usd": 0.50 - worst + 0.001}) + "\n")
        result = self.run_article()
        self.assertSafeFallback(result, "budget_exceeded")
        self.completion.assert_not_called()
        # just enough room for the full reservation: the call goes ahead
        self.ledger.write_text(json.dumps({"cost_usd": 0.50 - worst - 0.0001}) + "\n")
        mock.patch.object(llm_client, "_disabled_reason", None).start()
        self.assertEqual(self.run_article(raw_response(VALID_EDITS)).status, "rectified")

    def test_length_finish_is_reported_as_truncated(self):
        self.completion.side_effect = None
        self.completion.return_value = raw_response('{"edits": [', finish="length", completion=8192)
        with self.assertRaises(llm_client.LLMError) as ctx:
            llm_client.complete([{"role": "user", "content": "x" * 300}], 8192)
        self.assertEqual(ctx.exception.reason, "truncated")
        self.assertEqual(ctx.exception.usage.completion_tokens, 8192)

    def test_truncated_reply_is_never_applied_even_if_it_parses(self):
        # the reply happens to be valid JSON with a valid edit, but it stopped at the limit
        for content in (VALID_EDITS, VALID_EDITS[:60]):
            with self.subTest(content=content[:20]):
                self.completion.reset_mock()
                result = self.run_article(raw_response(content, finish="length", completion=8192))
                self.assertSafeFallback(result, "truncated")
                self.assertEqual(self.completion.call_count, 1)        # no repair or retry after truncation
                self.assertEqual(result.usage["completion_tokens"], 8192)
                self.assertFalse(any(Path(llm_client.CACHE_DIR).glob("*.json")))   # never cached

    def test_fallback_after_truncation_keeps_later_articles_working(self):
        self.assertSafeFallback(self.run_article(raw_response(VALID_EDITS, finish="length", completion=8192)),
                                "truncated")
        self.assertEqual(self.run_article(raw_response(VALID_EDITS)).status, "rectified")


if __name__ == "__main__":
    unittest.main()
