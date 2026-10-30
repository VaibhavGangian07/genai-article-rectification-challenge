"""
LiteLLM proxy budget enforcement. The proxy's read-only endpoints are faked;
no test sends a request anywhere.
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

import requests
import llm_client
from llm_client import LLMError

BASE = "https://proxy.example.test/"
KEY = "sk-proxy-test-key"
ENV = {"LLM_MODEL_NAME": "openai/gpt-oss-120b", "LLM_API_KEY": KEY, "LLM_API_BASE": BASE}
MESSAGES = [{"role": "user", "content": "x" * 300}]
# deliberately not Groq's rates, so any use of Groq pricing would show
PROXY_IN, PROXY_OUT = 1e-06, 4e-06


def reply():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, completion_tokens_details=None)
    choice = SimpleNamespace(message=SimpleNamespace(content='{"edits": []}'), finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage, _hidden_params={})


class FakeProxy:
    """Answers /v1/models, /model/info and /key/info like a LiteLLM proxy."""

    def __init__(self):
        self.models = ["gpt-oss-120b"]
        self.prices = {"input_cost_per_token": PROXY_IN, "output_cost_per_token": PROXY_OUT}
        self.spend, self.max_budget = 0.0, 3.0
        self.fail = set()
        self.calls = []

    def get(self, url, headers=None, timeout=None, **kwargs):
        self.calls.append({"url": url, "headers": headers, "kwargs": kwargs})
        path = url[len(BASE.rstrip("/")):]
        response = requests.Response()
        response.url = url
        if path in self.fail:
            response.status_code, response._content = 500, b"{}"
            return response
        payload = {
            "/v1/models": {"data": [{"id": m} for m in self.models]},
            "/model/info": {"data": [{"model_name": "gpt-oss-120b", "model_info": self.prices}]},
            "/key/info": {"key": "redacted", "info": {"spend": self.spend, "max_budget": self.max_budget}},
        }.get(path)
        response.status_code = 200 if payload is not None else 404
        response._content = json.dumps(payload or {}).encode()
        return response


class ProxyCase(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.addCleanup(mock.patch.stopall)
        self.ledger = Path(tmp.name) / "ledger.jsonl"
        mock.patch.object(llm_client, "LEDGER_PATH", self.ledger).start()
        mock.patch.object(llm_client, "CACHE_DIR", Path(tmp.name) / "cache").start()
        mock.patch.object(llm_client, "_disabled_reason", None).start()
        mock.patch.object(llm_client, "_proxy_state", {}).start()
        mock.patch.object(llm_client, "_sleep").start()
        mock.patch.dict(os.environ, ENV).start()
        os.environ.pop("MAX_SPEND_USD", None)
        self.proxy = FakeProxy()
        mock.patch.object(llm_client.requests, "get", side_effect=self.proxy.get).start()
        self.completion = mock.patch.object(llm_client, "_completion", return_value=reply()).start()

    def paths(self):
        return [c["url"][len(BASE.rstrip("/")):] for c in self.proxy.calls]

    def entries(self):
        return [json.loads(l) for l in self.ledger.read_text().splitlines()] if self.ledger.exists() else []


class TestProxyDetection(unittest.TestCase):

    def test_only_api_groq_com_is_direct_groq(self):
        self.assertFalse(llm_client.is_proxy("https://api.groq.com/openai/v1"))
        for base in ("https://recllm.brahmastra.tech/", "http://127.0.0.1:4000", "https://api.groq.com.evil.test/"):
            self.assertTrue(llm_client.is_proxy(base), base)

    def test_served_model_name(self):
        self.assertEqual(llm_client.served_model_name("openai/gpt-oss-120b"), "gpt-oss-120b")
        self.assertEqual(llm_client.served_model_name("gpt-oss-120b"), "gpt-oss-120b")


class TestProxyPricing(ProxyCase):

    def test_cost_uses_proxy_prices_not_groq(self):
        result = llm_client.complete(MESSAGES, max_tokens=1000)
        expected = 1000 * PROXY_IN + 500 * PROXY_OUT
        self.assertAlmostEqual(result.cost_usd, expected)
        self.assertNotAlmostEqual(result.cost_usd, 1000 * 0.15e-6 + 500 * 0.60e-6)
        [entry] = self.entries()
        self.assertTrue(entry["pricing_source"].startswith("https://proxy.example.test/model/info"))
        self.assertEqual(entry["proxy_spend_before"], 0.0)

    def test_pricing_is_verified_once_per_run(self):
        llm_client.complete(MESSAGES, max_tokens=1000)
        llm_client.complete(MESSAGES + [{"role": "user", "content": "another"}], max_tokens=1000)
        self.assertEqual(self.paths().count("/v1/models"), 1)
        self.assertEqual(self.paths().count("/model/info"), 1)

    def test_missing_or_invalid_prices_block_all_calls(self):
        for prices in ({}, {"input_cost_per_token": None, "output_cost_per_token": 6e-07},
                       {"input_cost_per_token": 0, "output_cost_per_token": 6e-07}):
            with self.subTest(prices=prices):
                mock.patch.object(llm_client, "_proxy_state", {}).start()
                mock.patch.object(llm_client, "_disabled_reason", None).start()
                self.proxy.prices = prices
                with self.assertRaises(LLMError) as ctx:
                    llm_client.complete(MESSAGES, max_tokens=1000)
                self.assertEqual(ctx.exception.reason, "pricing_unavailable")
                with self.assertRaises(LLMError) as again:
                    llm_client.complete(MESSAGES, max_tokens=1000)
                self.assertEqual(again.exception.reason, "disabled")
        self.completion.assert_not_called()

    def test_model_not_served_to_key_blocks_calls(self):
        self.proxy.models = ["some-other-model"]
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "model_unavailable")
        self.completion.assert_not_called()


class TestProxyBudget(ProxyCase):

    def test_key_info_is_read_before_every_attempt(self):
        err = type("RateLimitError", (Exception,), {})("slow down")
        err.status_code, err.litellm_response_headers = 429, {}
        order = []
        self.proxy_get = self.proxy.get
        def tracked_get(url, **kw):
            if url.endswith("/key/info"):
                order.append("key/info")
            return self.proxy_get(url, **kw)
        mock.patch.object(llm_client.requests, "get", side_effect=tracked_get).start()
        def completion(**kw):
            order.append("request")
            if order.count("request") < 3:
                raise err
            return reply()
        self.completion.side_effect = completion
        llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(order, ["key/info", "request"] * 3)

    def test_proxy_spend_near_hard_cap_blocks_the_call(self):
        self.proxy.spend = 2.9995
        with mock.patch.dict(os.environ, {"MAX_SPEND_USD": "3"}), self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")
        self.completion.assert_not_called()

    def test_key_max_budget_below_hard_cap_is_respected(self):
        self.proxy.spend, self.proxy.max_budget = 0.9995, 1.0
        with mock.patch.dict(os.environ, {"MAX_SPEND_USD": "3"}), self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")

    def test_key_max_budget_above_hard_cap_does_not_raise_it(self):
        self.proxy.spend, self.proxy.max_budget = 2.9995, 100.0
        with mock.patch.dict(os.environ, {"MAX_SPEND_USD": "3"}), self.assertRaises(LLMError):
            llm_client.complete(MESSAGES, max_tokens=1000)

    def test_phase_cap_applies_even_when_the_account_has_room(self):
        self.ledger.write_text(json.dumps({"cost_usd": 0.4999}) + "\n")
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_exceeded")
        self.assertIn("phase spend $0.4999", str(ctx.exception))
        self.completion.assert_not_called()

    def test_delayed_proxy_spend_is_covered_by_the_ledger(self):
        self.proxy.spend = 1.0
        self.assertEqual(llm_client._effective_proxy_spend(1.0), 1.0)
        self.ledger.write_text(json.dumps({"cost_usd": 0.5}) + "\n")      # our calls, not yet counted by the proxy
        self.assertEqual(llm_client._effective_proxy_spend(1.0), 1.5)
        self.assertEqual(llm_client._effective_proxy_spend(1.7), 1.7)      # proxy caught up and then some

    def test_unreadable_budget_blocks_calls(self):
        for broken in ("/key/info",):
            self.proxy.fail.add(broken)
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_unverifiable")
        self.assertIn("HTTP 500", str(ctx.exception))
        self.completion.assert_not_called()
        with self.assertRaises(LLMError) as again:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(again.exception.reason, "disabled")

    def test_network_error_blocks_calls_without_exposing_details(self):
        mock.patch.object(llm_client.requests, "get",
                          side_effect=requests.ConnectionError(f"failed {BASE}key/info?key={KEY}")).start()
        with self.assertRaises(LLMError) as ctx:
            llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertEqual(ctx.exception.reason, "budget_unverifiable")
        self.assertNotIn(KEY, str(ctx.exception))
        self.completion.assert_not_called()


class TestCredentialHandling(ProxyCase):

    def test_key_only_in_authorization_header(self):
        llm_client.complete(MESSAGES, max_tokens=1000)
        self.assertTrue(self.proxy.calls)
        for call in self.proxy.calls:
            self.assertNotIn(KEY, call["url"])
            self.assertNotIn("params", call["kwargs"])
            self.assertEqual(call["headers"], {"Authorization": f"Bearer {KEY}"})
        self.assertNotIn(KEY, self.ledger.read_text())

    def test_model_sent_to_litellm_unchanged(self):
        llm_client.complete(MESSAGES, max_tokens=1000)
        kwargs = self.completion.call_args.kwargs
        self.assertEqual((kwargs["model"], kwargs["api_base"]), ("openai/gpt-oss-120b", BASE))

    def test_cache_hit_needs_no_proxy_access(self):
        llm_client.complete(MESSAGES, max_tokens=1000)
        calls = len(self.proxy.calls)
        self.proxy.fail.update({"/key/info", "/v1/models", "/model/info"})
        self.assertTrue(llm_client.complete(MESSAGES, max_tokens=1000).cached)
        self.assertEqual(len(self.proxy.calls), calls)


if __name__ == "__main__":
    unittest.main()
