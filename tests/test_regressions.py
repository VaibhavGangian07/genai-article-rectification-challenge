"""
Regression tests for the Phase 1 integration review: duplicate execution,
status accounting, fallback behaviour, usage aggregation and per-attempt
budget checks. No test reaches a real provider.
"""

import io
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
import rectifier
import rectification_system
from llm_client import LLMError, LLMResponse
from article_utils import strip_annotations, find_annotation_leak

ENV = {"LLM_MODEL_NAME": "groq/openai/gpt-oss-120b", "LLM_API_KEY": "dummy-test-key", "LLM_API_BASE": "http://127.0.0.1:9"}
MESSAGES = [{"role": "user", "content": "x" * 300}]
PRICE = llm_client.PRICING[ENV["LLM_MODEL_NAME"]]
MAPPING = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}
AI_TEXT = "## Title\nThe team won 3 matches in 2024.\n\n**Error Annotations:**\n[\n  {\"error\": \"won 3\"}\n]"
BODY = strip_annotations(AI_TEXT)


def reply(edits, **kw):
    text = edits if isinstance(edits, str) else json.dumps({"edits": edits})
    fields = dict(text=text, finish_reason="stop", prompt_tokens=1000, completion_tokens=200,
                  reasoning_tokens=100, cost_usd=0.00027)
    fields.update(kw)
    return LLMResponse(**fields)


def raw_response(choices="default", usage="default"):
    """Objects shaped like litellm's ModelResponse, with optional parts missing."""
    if choices == "default":
        choices = [SimpleNamespace(message=SimpleNamespace(content='{"edits": []}'), finish_reason="stop")]
    if usage == "default":
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500,
                                completion_tokens_details=SimpleNamespace(reasoning_tokens=300))
    fields = {"choices": choices, "_hidden_params": {}}
    if usage is not None:
        fields["usage"] = usage
    return SimpleNamespace(**fields)


def error(name, status):
    e = type(name, (Exception,), {})(f"fake {name}")
    e.status_code, e.litellm_response_headers = status, {}
    return e


class ClientCase(unittest.TestCase):

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

    def entries(self):
        return [json.loads(l) for l in self.ledger.read_text().splitlines()] if self.ledger.exists() else []


class TestResponseShapes(ClientCase):
    """Responses missing optional fields must not crash and must not be recorded as free."""

    def call(self, response):
        self.completion.side_effect = None
        self.completion.return_value = response
        return llm_client.complete(MESSAGES, max_tokens=1000)

    def test_no_choices(self):
        result = self.call(raw_response(choices=[]))
        self.assertEqual((result.text, result.finish_reason), ("", ""))
        self.assertEqual(len(self.entries()), 1)

    def test_null_content_and_missing_details(self):
        choices = [SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")]
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, completion_tokens_details=None)
        result = self.call(raw_response(choices=choices, usage=usage))
        self.assertEqual((result.text, result.reasoning_tokens), ("", 0))
        self.assertAlmostEqual(result.cost_usd, 0.00045)

    def test_missing_or_zero_usage_is_charged_at_worst_case(self):
        worst = llm_client._worst_case(PRICE, MESSAGES, 1000)
        for n, usage in enumerate((None, SimpleNamespace(prompt_tokens=0, completion_tokens=0),
                                   SimpleNamespace(prompt_tokens=None, completion_tokens=None))):
            with self.subTest(usage=usage):
                self.ledger.unlink(missing_ok=True)
                mock.patch.object(llm_client, "CACHE_DIR", self.ledger.parent / f"cache{n}").start()
                result = self.call(raw_response(usage=usage))
                self.assertAlmostEqual(result.cost_usd, worst)
                self.assertEqual(self.entries()[-1]["cost_source"], "worst_case")

    def test_failed_cache_write_keeps_the_reply(self):
        with mock.patch.object(llm_client, "CACHE_DIR", Path("Z:/no/such/drive/\0invalid")):
            result = self.call(raw_response())
        self.assertEqual(result.text, '{"edits": []}')


class TestUsageAcrossAttempts(ClientCase):

    def test_retry_costs_are_included_in_the_reply(self):
        worst = llm_client._worst_case(PRICE, MESSAGES, 1000)
        self.completion.side_effect = [error("Timeout", 408), raw_response()]
        result = llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(result.attempts, 2)
        self.assertAlmostEqual(result.cost_usd, worst + 0.00045)
        self.assertAlmostEqual(llm_client.spent(), result.cost_usd)

    def test_failed_request_carries_its_cost_and_attempts(self):
        worst = llm_client._worst_case(PRICE, MESSAGES, 1000)
        self.completion.side_effect = [error("Timeout", 408)] * 3
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.usage.attempts, 3)
        self.assertAlmostEqual(ctx.exception.usage.cost_usd, 3 * worst)

    def test_truncated_reply_carries_tokens_and_cost(self):
        self.completion.side_effect = None
        self.completion.return_value = raw_response(choices=[
            SimpleNamespace(message=SimpleNamespace(content='{"edits": ['), finish_reason="length")])
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        usage = ctx.exception.usage
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens, usage.attempts), (1000, 500, 1))
        self.assertAlmostEqual(usage.cost_usd, 0.00045)

    def test_budget_refusal_reports_no_attempt(self):
        self.ledger.write_text(json.dumps({"cost_usd": 0.4999}) + "\n")
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual((ctx.exception.usage.attempts, ctx.exception.usage.cost_usd), (0, 0.0))

    def test_cache_hit_is_not_counted_as_a_request(self):
        self.completion.side_effect = None
        self.completion.return_value = raw_response()
        llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(llm_client.complete(MESSAGES, max_tokens=1000).attempts, 0)


class TestBudgetBeforeEveryAttempt(ClientCase):

    def test_spend_is_checked_before_each_request(self):
        order = []
        real_spent = llm_client.spent
        mock.patch.object(llm_client, "spent", side_effect=lambda: order.append("check") or real_spent()).start()
        def fake_completion(**kwargs):
            order.append("request")
            if order.count("request") < 3:
                raise error("RateLimitError", 429)
            return raw_response()
        self.completion.side_effect = fake_completion
        llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(order, ["check", "request", "check", "request", "check", "request"])


class EngineCase(unittest.TestCase):

    def setUp(self):
        self.complete = mock.patch.object(llm_client, "complete",
                                          side_effect=AssertionError("unexpected API call")).start()
        self.addCleanup(mock.patch.stopall)


class TestEngineCalls(EngineCase):

    def test_valid_reply_needs_exactly_one_request(self):
        self.complete.side_effect = [reply([])]
        rectification_system.run(AI_TEXT, "Source.", "a")
        self.assertEqual(self.complete.call_count, 1)

    def test_at_most_one_json_repair(self):
        self.complete.side_effect = [reply("bad"), reply("still bad"), reply([])]
        result = rectification_system.run(AI_TEXT, "Source.", "a")
        self.assertEqual((self.complete.call_count, result.status), (2, "fallback"))

    def test_usage_includes_failed_and_repair_requests(self):
        truncated = LLMError("truncated", "", reply("", completion_tokens=4096, cost_usd=0.0026))
        self.complete.side_effect = [reply("bad"), truncated]
        result = rectification_system.run(AI_TEXT, "Source.", "a")
        self.assertEqual(result.status, "fallback")
        self.assertEqual((result.usage["calls"], result.usage["completion_tokens"]), (2, 4296))
        self.assertAlmostEqual(result.usage["cost_usd"], 0.00287)


class TestEngineStatus(EngineCase):

    EDIT = {"op": "replace", "find": "won 3 matches", "replace": "won 5 matches",
            "evidence": "The team won 5 matches in 2024, officials said."}

    def test_noop_edit_is_not_a_correction(self):
        body = 'He said \u201cyes\u201d at the club in 2024.'
        noop = {"op": "replace", "find": 'He said "yes" at', "replace": "He said \u201cyes\u201d at", "evidence": "a b c"}
        self.complete.side_effect = [reply([noop])]
        result = rectification_system.run(body, "Source.", "a")
        self.assertEqual((result.status, result.text), ("no_valid_edits", body))
        self.assertEqual(result.skipped, [{"find": noop["find"], "reason": "noop"}])

    def test_validation_error_skips_only_that_edit(self):
        other = {"op": "replace", "find": "in 2024", "replace": "in 2025", "evidence": "x y z"}
        self.complete.side_effect = [reply([other, self.EDIT])]
        real = rectification_system.check_edit
        def flaky(edit, *args):
            if edit.find == "in 2024":
                raise IndexError("boom")
            return real(edit, *args)
        with mock.patch.object(rectification_system, "check_edit", side_effect=flaky):
            result = rectification_system.run(AI_TEXT, "The team won 5 matches in 2024, officials said.", "a")
        self.assertEqual(result.status, "rectified")
        self.assertEqual([a["find"] for a in result.applied], ["won 3 matches"])
        self.assertEqual(result.skipped, [{"find": "in 2024", "reason": "validation_error: IndexError"}])

    def test_every_fallback_on_every_article_is_the_stripped_body(self):
        for article_id, entry in MAPPING.items():
            ai = (ROOT / entry["ai_generated_file"]).read_text(encoding="utf-8")
            self.complete.side_effect = [LLMError("llm_error:Timeout")]
            result = rectification_system.run(ai, "Source.", article_id)
            with self.subTest(article=article_id):
                self.assertEqual((result.status, result.text), ("fallback", strip_annotations(ai)))
                self.assertIsNone(find_annotation_leak(result.text))


class BatchCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        self.tmp = Path(tmp.name)
        (self.tmp / "ai.txt").write_text(AI_TEXT, encoding="utf-8")
        (self.tmp / "src.txt").write_text("Source.", encoding="utf-8")
        self.mapping = {f"article_9{i:02d}": {"article_id": f"article_9{i:02d}", "ai_generated_file": str(self.tmp / "ai.txt"),
                                              "source_file": str(self.tmp / "src.txt"),
                                              "rectified_file": str(self.tmp / "out" / f"{i}.txt")} for i in range(6)}
        self.mapping["article_905"]["ai_generated_file"] = str(self.tmp / "missing.txt")
        mock.patch.object(rectifier, "get_article_mapping", side_effect=lambda a: self.mapping[a]).start()
        self.run_mock = mock.patch.object(rectifier, "run", side_effect=AssertionError("unexpected engine call")).start()
        self.stdout = io.StringIO()
        mock.patch.object(sys, "stdout", self.stdout).start()
        mock.patch.object(sys, "stderr", io.StringIO()).start()

    def output(self, article_id):
        return Path(self.mapping[article_id]["rectified_file"]).read_text(encoding="utf-8")


class TestBatchAccounting(BatchCase):

    RESULTS = {
        "article_900": SimpleNamespace(text="Fixed.", status="rectified", reason="",
                                       usage={"calls": 1, "prompt_tokens": 1000, "completion_tokens": 200, "cost_usd": 0.0003}),
        "article_901": SimpleNamespace(text=BODY, status="no_change", reason="", usage={"calls": 1, "cost_usd": 0.0002}),
        "article_902": SimpleNamespace(text=BODY, status="no_valid_edits", reason="", usage={"calls": 2, "cost_usd": 0.0005}),
        "article_903": SimpleNamespace(text=BODY, status="fallback", reason="truncated", usage={"calls": 1, "cost_usd": 0.0026}),
        "article_904": SimpleNamespace(text="Done.", status="finished", reason="", usage={}),
    }

    def test_each_article_has_exactly_one_status(self):
        self.run_mock.side_effect = lambda ai, src, article_id: self.RESULTS[article_id]
        outcomes = rectifier.process_articles([{"article_id": a} for a in self.mapping])
        self.assertEqual(outcomes, {"rectified": ["article_900"], "no_change": ["article_901"],
                                    "no_valid_edits": ["article_902"], "fallback": ["article_903", "article_904"],
                                    "failed": ["article_905"]})
        self.assertEqual(sorted(a for ids in outcomes.values() for a in ids), sorted(self.mapping))
        self.assertEqual(self.output("article_904"), BODY)   # unknown status falls back to the stripped text
        summary = self.stdout.getvalue()
        self.assertIn("Rectified:      1", summary)
        self.assertIn("Fallback:       2", summary)
        self.assertIn("article_904: ValueError: rectifier returned unknown status 'finished'", summary)
        self.assertIn("LLM usage:      5 requests, 1000 prompt + 200 completion tokens, $0.0036 recorded", summary)

    def test_each_article_is_rectified_once(self):
        self.run_mock.side_effect = lambda ai, src, article_id: self.RESULTS["article_900"]
        ids = ["article_900", "article_901", "article_900", "article_902", "article_901"]
        outcomes = rectifier.process_articles([{"article_id": a} for a in ids])
        self.assertEqual([c.args[2] for c in self.run_mock.call_args_list], ["article_900", "article_901", "article_902"])
        self.assertEqual(outcomes["rectified"], ["article_900", "article_901", "article_902"])
        self.assertIn("Skipping 2 duplicate article entries", self.stdout.getvalue())

    def test_rectify_article_calls_the_engine_once(self):
        self.run_mock.side_effect = None
        self.run_mock.return_value = self.RESULTS["article_900"]
        self.assertEqual(rectifier.rectify_article("article_900"), "Fixed.")
        self.assertEqual(self.run_mock.call_count, 1)


class TestBatchFallback(BatchCase):

    def test_engine_and_rectifier_fallbacks_save_the_same_stripped_text(self):
        engine_fallback = SimpleNamespace(text=BODY, status="fallback", reason="budget_exceeded", usage={})
        cases = {"article_900": engine_fallback, "article_901": RuntimeError("crash"),
                 "article_902": SimpleNamespace(text=AI_TEXT, status="rectified", reason="", usage={})}
        def fake(ai, src, article_id):
            outcome = cases[article_id]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        self.run_mock.side_effect = fake
        outcomes = rectifier.process_articles([{"article_id": a} for a in cases])
        self.assertEqual(outcomes["fallback"], list(cases))
        for article_id in cases:
            with self.subTest(article=article_id):
                self.assertEqual(self.output(article_id), BODY)
                self.assertIsNone(find_annotation_leak(self.output(article_id)))


if __name__ == "__main__":
    unittest.main()
