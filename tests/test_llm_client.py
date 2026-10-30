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
from llm_client import LLMError

MESSAGES = [{"role": "user", "content": "x" * 300}]
ENV = {"LLM_MODEL_NAME": "groq/openai/gpt-oss-120b", "LLM_API_KEY": "dummy-test-key",
       "LLM_API_BASE": "http://127.0.0.1:9/openai/v1"}


def response(content='{"edits": []}', finish="stop", prompt=1000, completion=500, reasoning=300):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning))
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)],
                           usage=usage, _hidden_params={"response_cost": 0.00045})


def error(name, status=None, retry_after=None):
    """An exception shaped like litellm's: class name, status_code and response headers."""
    cls = type(name, (Exception,), {})
    e = cls(f"fake {name}")
    e.status_code = status
    e.litellm_response_headers = {"retry-after": retry_after} if retry_after else {}
    return e


class LLMClientTestCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.ledger = Path(tmp.name) / "ledger.jsonl"
        for name, value in (("LEDGER_PATH", self.ledger), ("CACHE_DIR", Path(tmp.name) / "cache"),
                            ("_disabled_reason", None)):
            p = mock.patch.object(llm_client, name, value)
            p.start()
            self.addCleanup(p.stop)
        env = mock.patch.dict(os.environ, ENV)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("MAX_SPEND_USD", None)
        self.sleep = mock.patch.object(llm_client, "_sleep").start()
        self.addCleanup(mock.patch.stopall)
        self.completion = mock.patch.object(llm_client, "_completion",
                                            side_effect=AssertionError("unexpected API call")).start()

    def entries(self):
        if not self.ledger.exists():
            return []
        return [json.loads(line) for line in self.ledger.read_text().splitlines()]

    def seed_spend(self, amount):
        self.ledger.write_text(json.dumps({"cost_usd": amount}) + "\n")


class TestSuccessfulCall(LLMClientTestCase):

    def test_cost_usage_and_ledger(self):
        self.completion.side_effect = None
        self.completion.return_value = response()
        result = llm_client.complete(MESSAGES, max_tokens=1000, article_id="article_001")
        self.assertEqual((result.text, result.prompt_tokens, result.completion_tokens, result.reasoning_tokens),
                         ('{"edits": []}', 1000, 500, 300))
        self.assertAlmostEqual(result.cost_usd, 1000 * 0.15e-6 + 500 * 0.60e-6)
        [entry] = self.entries()
        self.assertEqual((entry["outcome"], entry["article_id"], entry["cost_source"]), ("ok", "article_001", "pricing"))
        self.assertAlmostEqual(entry["cost_usd"], 0.00045)
        self.assertEqual(entry["litellm_cost_usd"], 0.00045)
        self.assertNotIn("dummy-test-key", self.ledger.read_text())

    def test_request_parameters(self):
        self.completion.side_effect = None
        self.completion.return_value = response()
        llm_client.complete(MESSAGES, max_tokens=1234)
        kwargs = self.completion.call_args.kwargs
        self.assertEqual(kwargs["model"], "groq/openai/gpt-oss-120b")
        self.assertEqual(kwargs["api_base"], ENV["LLM_API_BASE"])
        self.assertEqual((kwargs["max_tokens"], kwargs["temperature"], kwargs["reasoning_effort"]), (1234, 0, "medium"))
        self.assertEqual(kwargs["response_format"], {"type": "json_object"})
        self.assertEqual((kwargs["num_retries"], kwargs["max_retries"]), (0, 0))

    def test_cached_reply_costs_nothing(self):
        self.completion.side_effect = None
        self.completion.return_value = response()
        llm_client.complete(MESSAGES, max_tokens=1000)
        again = llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(self.completion.call_count, 1)
        self.assertEqual((again.cached, again.cost_usd, again.text), (True, 0.0, '{"edits": []}'))
        self.assertEqual(len(self.entries()), 1)

    def test_truncated_reply_raises_and_is_not_cached(self):
        self.completion.side_effect = None
        self.completion.return_value = response(content='{"edits": [', finish="length")
        for _ in range(2):
            with self.assertRaises(LLMError) as ctx:
                llm_client.complete(MESSAGES, max_tokens=1000)
            self.assertEqual(ctx.exception.reason, "truncated")
        self.assertEqual(self.completion.call_count, 2)
        self.assertEqual([e["finish_reason"] for e in self.entries()], ["length", "length"])


class TestRetries(LLMClientTestCase):

    def test_rate_limit_retried_with_retry_after(self):
        self.completion.side_effect = [error("RateLimitError", 429, retry_after="7"), response()]
        result = llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(result.text, '{"edits": []}')
        self.sleep.assert_called_once_with(7.0)
        self.assertEqual([e["outcome"] for e in self.entries()], ["error:RateLimitError", "ok"])

    def test_server_errors_and_timeouts_retried_at_most_twice(self):
        for err in (error("InternalServerError", 500), error("BadGatewayError", 502), error("Timeout", 408)):
            with self.subTest(error=type(err).__name__):
                self.completion.reset_mock()
                self.completion.side_effect = [err] * 5
                with self.assertRaises(LLMError):
                    llm_client.complete(MESSAGES, max_tokens=1000)
                self.assertEqual(self.completion.call_count, 3)

    def test_non_recoverable_errors_are_not_retried(self):
        for err in (error("BadRequestError", 400), error("APIConnectionError", 500)):
            with self.subTest(error=type(err).__name__):
                self.completion.reset_mock()
                self.completion.side_effect = [err, response()]
                with self.assertRaises(LLMError) as ctx:
                    llm_client.complete(MESSAGES, max_tokens=1000)
                self.assertEqual(ctx.exception.reason, f"llm_error:{type(err).__name__}")
                self.assertEqual(self.completion.call_count, 1)

    def test_authentication_error_stops_all_further_calls(self):
        self.completion.side_effect = [error("AuthenticationError", 401), response()]
        with self.assertRaises(LLMError):
            llm_client.complete(MESSAGES, max_tokens=1000)
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "disabled")
        self.assertEqual(self.completion.call_count, 1)

    def test_unknown_billing_is_charged_at_worst_case(self):
        self.completion.side_effect = [error("APIConnectionError", 500)]
        with self.assertRaises(LLMError):
            llm_client.complete(MESSAGES, max_tokens=1000)
        [entry] = self.entries()
        self.assertEqual(entry["cost_source"], "worst_case")
        self.assertGreater(entry["cost_usd"], 1000 * 0.60e-6)


class TestBudget(LLMClientTestCase):

    def test_default_cap_and_hard_cap(self):
        self.assertEqual(llm_client.spend_cap(), 0.50)
        for value, expected in (("1.25", 1.25), ("10", 3.00), ("abc", 0.50), ("-1", 0.0)):
            with self.subTest(value=value), mock.patch.dict(os.environ, {"MAX_SPEND_USD": value}):
                self.assertEqual(llm_client.spend_cap(), expected)

    def test_call_refused_when_cap_would_be_exceeded(self):
        self.seed_spend(0.4999)
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")
        self.completion.assert_not_called()

    def test_hard_cap_applies_even_with_a_higher_configured_cap(self):
        self.seed_spend(2.9999)
        with mock.patch.dict(os.environ, {"MAX_SPEND_USD": "50"}), self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")
        self.completion.assert_not_called()

    def test_every_retry_passes_the_budget_check(self):
        # The first attempt fits; its timeout is charged at worst case, so the retry no longer fits
        worst = llm_client._worst_case(llm_client.PRICING[ENV["LLM_MODEL_NAME"]], MESSAGES, 1000)
        self.seed_spend(0.50 - worst * 1.5)
        self.completion.side_effect = [error("Timeout", 408), response()]
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")
        self.assertEqual(self.completion.call_count, 1)

    def test_unpriced_model_is_never_called(self):
        with mock.patch.dict(os.environ, {"LLM_MODEL_NAME": "openai/gpt-oss-120b"}), self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "pricing_unavailable")
        self.completion.assert_not_called()

    def test_missing_key_is_never_called(self):
        with mock.patch.dict(os.environ, {"LLM_API_KEY": ""}), self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "missing_config")
        self.completion.assert_not_called()


class TestOfflineConfiguration(unittest.TestCase):

    def test_litellm_uses_bundled_price_map_and_tokenizer(self):
        self.assertEqual(os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP"), "True")
        tokenizer_dir = Path(os.environ["CUSTOM_TIKTOKEN_CACHE_DIR"])
        self.assertTrue((tokenizer_dir / "9b5ad71b2ce5302211f9c61530b329a4922fc6a4").is_file())


if __name__ == "__main__":
    unittest.main()
