"""
The single place that calls the LLM provider.

Every attempt is checked against the spending caps first and recorded in a
local ledger afterwards.

- Direct Groq (api.groq.com): no balance endpoint exists, so the local ledger
  is the record of spend, priced at Groq's published rates.
- LiteLLM proxy (any other base URL): prices come from the proxy's own
  /model/info, the model must be listed by /v1/models, and the key's spend and
  max budget are read from /key/info before every attempt. Groq's rates are
  never assumed for a proxy. If any of this cannot be verified, no call is made.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import requests
from dotenv import load_dotenv

REPO_DIR = Path(__file__).resolve().parent

# Keep litellm offline apart from the provider call: use its bundled price
# map, and the tokenizer file shipped in this repo instead of downloading it.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("CUSTOM_TIKTOKEN_CACHE_DIR", str(REPO_DIR / "tokenizers"))

load_dotenv()

HARD_CAP_USD = 3.00            # assignment limit; cannot be raised by configuration
DEFAULT_SPEND_CAP_USD = 0.50   # experimental cap, override with MAX_SPEND_USD (clamped to the hard cap)

# Official Groq pricing, https://console.groq.com/docs/model/openai/gpt-oss-120b (retrieved 2026-10-30)
PRICING = {"groq/openai/gpt-oss-120b": {"input_per_m": 0.15, "output_per_m": 0.60}}
PRICING_SOURCE = "https://console.groq.com/docs/model/openai/gpt-oss-120b (2026-10-30)"

LEDGER_PATH = REPO_DIR / ".llm_usage" / "ledger.jsonl"
CACHE_DIR = REPO_DIR / ".llm_cache"

GROQ_HOST = "api.groq.com"

REASONING_EFFORT = "medium"
TIMEOUT_SECONDS = 120
PROXY_TIMEOUT_SECONDS = 15
MAX_ATTEMPTS = 3               # first try plus two retries, for 429 / timeout / 5xx only
MAX_BACKOFF_SECONDS = 30

_sleep = time.sleep
_disabled_reason = None        # set after a non-recoverable error; stops further calls
_proxy_state = {}              # verified proxy pricing and the first spend reading of this run


@dataclass
class LLMResponse:
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float            # everything recorded in the ledger for this request, retries included
    cached: bool = False
    attempts: int = 1          # provider requests made (0 for a cache hit)


class LLMError(Exception):
    """
    A call failed; reason is a short code used in reports. usage holds the
    tokens (if a reply arrived) and the cost recorded for the failed request.
    """
    def __init__(self, reason: str, message: str = "", usage: LLMResponse = None):
        super().__init__(f"{reason}: {message}" if message else reason)
        self.reason = reason
        self.usage = usage


def spend_cap() -> float:
    """The experimental cap from MAX_SPEND_USD, never above HARD_CAP_USD."""
    try:
        cap = float(os.getenv("MAX_SPEND_USD", DEFAULT_SPEND_CAP_USD))
    except ValueError:
        cap = DEFAULT_SPEND_CAP_USD
    return min(max(cap, 0.0), HARD_CAP_USD)


def spent() -> float:
    """Total cost recorded in the ledger."""
    if not LEDGER_PATH.exists():
        return 0.0
    with open(LEDGER_PATH, "r", encoding="utf-8") as f:
        return sum(json.loads(line).get("cost_usd", 0.0) for line in f if line.strip())


def _record(entry: dict):
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _cost(price: dict, prompt_tokens: int, completion_tokens: int) -> float:
    return prompt_tokens * price["input_per_m"] / 1e6 + completion_tokens * price["output_per_m"] / 1e6


def _worst_case(price: dict, messages: list, max_tokens: int) -> float:
    # ~3 characters per token overestimates the prompt; assume every output token is used
    prompt_chars = sum(len(m["content"]) for m in messages)
    return _cost(price, prompt_chars // 3 + 1, max_tokens)


def _completion(**kwargs):
    import litellm
    return litellm.completion(**kwargs)


def _is_transient(error) -> bool:
    name = type(error).__name__
    if name == "Timeout":
        return True
    if name == "APIConnectionError":   # also raised for unparseable responses; not retried
        return False
    status = getattr(error, "status_code", None)
    return status == 429 or (isinstance(status, int) and 500 <= status < 600)


def _billing_unknown(error) -> bool:
    """The request may have been processed even though no usage came back."""
    return type(error).__name__ in ("Timeout", "APIConnectionError")


def _backoff(error, attempt: int) -> float:
    headers = getattr(error, "litellm_response_headers", None) or {}
    try:
        delay = float(dict(headers).get("retry-after"))
    except (TypeError, ValueError):
        delay = 2 ** attempt
    return min(delay, MAX_BACKOFF_SECONDS)


def _cache_path(payload: dict) -> Path:
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def is_proxy(api_base) -> bool:
    """Anything other than direct Groq is treated as a LiteLLM proxy and must pass proxy verification."""
    return bool(api_base) and urlparse(api_base).hostname != GROQ_HOST


def served_model_name(model: str) -> str:
    """litellm's openai/ adapter strips the prefix, so the proxy receives the bare name."""
    return model.split("/", 1)[1] if model.startswith("openai/") else model


def _proxy_get(api_base: str, api_key: str, path: str) -> dict:
    """Authenticated read-only GET; the key goes only in the Authorization header, never in the URL."""
    try:
        response = requests.get(api_base.rstrip("/") + path, headers={"Authorization": f"Bearer {api_key}"},
                                timeout=PROXY_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        # the exception text is deliberately not included: it can contain request details
        raise LLMError("budget_unverifiable",
                       f"GET {path} failed: {type(e).__name__}" + (f" (HTTP {status})" if status else "")) from None


def _proxy_pricing(api_base: str, api_key: str, model: str) -> dict:
    """The proxy's own per-token prices for the configured model, verified once per run."""
    if "pricing" not in _proxy_state:
        name = served_model_name(model)
        served = [m.get("id") for m in _proxy_get(api_base, api_key, "/v1/models").get("data", [])]
        if name not in served:
            raise LLMError("model_unavailable", f"proxy does not serve {name!r} to this key (serves {served})")
        entry = next((m for m in _proxy_get(api_base, api_key, "/model/info").get("data", [])
                      if m.get("model_name") == name), None)
        info = (entry or {}).get("model_info") or {}
        cost_in, cost_out = info.get("input_cost_per_token"), info.get("output_cost_per_token")
        if not all(isinstance(c, (int, float)) and c > 0 for c in (cost_in, cost_out)):
            raise LLMError("pricing_unavailable", f"proxy /model/info gives no prices for {name!r}")
        _proxy_state["pricing"] = {
            "input_per_m": cost_in * 1e6, "output_per_m": cost_out * 1e6,
            "source": f"{api_base.rstrip('/')}/model/info ({datetime.now(timezone.utc).date().isoformat()})",
        }
    return _proxy_state["pricing"]


def _proxy_spend(api_base: str, api_key: str):
    """(spend recorded by the proxy for this key, the key's max budget or None)."""
    data = _proxy_get(api_base, api_key, "/key/info")
    info = data.get("info", data)
    spend, max_budget = info.get("spend"), info.get("max_budget")
    if not isinstance(spend, (int, float)):
        raise LLMError("budget_unverifiable", "/key/info reports no spend")
    return float(spend), (float(max_budget) if isinstance(max_budget, (int, float)) else None)


def _effective_proxy_spend(proxy_spend: float) -> float:
    """
    The proxy may record spend with a delay, so never count less than its
    first reading this run plus everything our ledger has recorded since.
    """
    if "baseline" not in _proxy_state:
        _proxy_state["baseline"] = (proxy_spend, spent())
    base_proxy, base_ledger = _proxy_state["baseline"]
    return max(proxy_spend, base_proxy + spent() - base_ledger)


def complete(messages: list, max_tokens: int, article_id: str = "") -> LLMResponse:
    """
    Send one chat request and return the reply.

    Raises:
        LLMError: with reason 'disabled', 'missing_config', 'pricing_unavailable',
        'model_unavailable', 'budget_unverifiable', 'budget_exceeded', 'truncated', 'cache_miss',
        or 'llm_error:<type>'.
    """
    global _disabled_reason
    if _disabled_reason:
        raise LLMError("disabled", _disabled_reason)

    model, api_key, api_base = os.getenv("LLM_MODEL_NAME"), os.getenv("LLM_API_KEY"), os.getenv("LLM_API_BASE")
    if not model or not api_key:
        _disabled_reason = "LLM_MODEL_NAME or LLM_API_KEY not set"
        raise LLMError("missing_config", _disabled_reason)

    params = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
              "reasoning_effort": REASONING_EFFORT, "response_format": {"type": "json_object"}}
    cache_file = _cache_path(params)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return LLMResponse(**{**cached, "cost_usd": 0.0, "cached": True, "attempts": 0})
    if os.getenv("LLM_CACHE_ONLY") == "1":   # reproduce a run from its cached replies; never contact the provider
        raise LLMError("cache_miss", "LLM_CACHE_ONLY=1 and no cached reply for this request")

    proxy = is_proxy(api_base)
    try:
        if proxy:
            price = _proxy_pricing(api_base, api_key, model)
            pricing_source = price["source"]
        else:
            price, pricing_source = PRICING.get(model), PRICING_SOURCE
            if price is None:
                raise LLMError("pricing_unavailable", f"no verified pricing for model {model}")
    except LLMError as e:
        _disabled_reason = str(e)
        raise

    cap = spend_cap()
    charged, sent = 0.0, 0

    def failure(reason, message, reply=None):
        usage = reply or LLMResponse("", "", 0, 0, 0, 0.0)
        usage.cost_usd, usage.attempts = charged, sent
        return LLMError(reason, message, usage)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        worst = _worst_case(price, messages, max_tokens)
        total = spent()                      # this phase's spend, from our ledger
        account, hard_limit, proxy_spend = total, HARD_CAP_USD, None
        if proxy:
            try:
                proxy_spend, max_budget = _proxy_spend(api_base, api_key)
            except LLMError as e:
                _disabled_reason = str(e)
                raise failure(e.reason, str(e))
            account = _effective_proxy_spend(proxy_spend)
            if max_budget is not None:
                hard_limit = min(hard_limit, max_budget)
        if total + worst > cap or account + worst > hard_limit:
            _disabled_reason = (f"phase spend ${total:.4f} (cap ${cap:.2f}), account spend ${account:.4f} "
                                f"(limit ${hard_limit:.2f}), next call could cost ${worst:.4f}")
            raise failure("budget_exceeded", _disabled_reason)

        entry = {"time": datetime.now(timezone.utc).isoformat(), "article_id": article_id,
                 "model": model, "attempt": attempt, "pricing_source": pricing_source}
        if proxy:
            entry["proxy_spend_before"] = proxy_spend
        sent += 1
        try:
            response = _completion(**params, api_key=api_key, api_base=api_base,
                                   timeout=TIMEOUT_SECONDS, num_retries=0, max_retries=0)
        except Exception as e:
            cost = worst if _billing_unknown(e) else 0.0
            charged += cost
            entry.update(outcome=f"error:{type(e).__name__}", status_code=getattr(e, "status_code", None),
                         cost_usd=cost, cost_source="worst_case" if _billing_unknown(e) else "none")
            _record(entry)
            if _is_transient(e) and attempt < MAX_ATTEMPTS:
                _sleep(_backoff(e, attempt))
                continue
            if getattr(e, "status_code", None) in (401, 403, 404):
                _disabled_reason = f"{type(e).__name__} (status {e.status_code})"
            raise failure(f"llm_error:{type(e).__name__}", str(e)[:200]) from e

        result, cost_source = _read_response(response, price, worst)
        charged += result.cost_usd
        entry.update(outcome="ok", finish_reason=result.finish_reason, prompt_tokens=result.prompt_tokens,
                     completion_tokens=result.completion_tokens, reasoning_tokens=result.reasoning_tokens,
                     cost_usd=result.cost_usd, cost_source=cost_source,
                     litellm_cost_usd=(getattr(response, "_hidden_params", None) or {}).get("response_cost"))
        _record(entry)
        result.cost_usd, result.attempts = charged, sent

        if result.finish_reason == "length":
            raise failure("truncated", f"stopped at max_tokens={max_tokens}", result)
        if result.text:
            _write_cache(cache_file, result)
        return result


def _read_response(response, price: dict, worst: float):
    """
    Extract the reply and its usage. Fields may be missing: litellm reports
    absent token counts as 0, so a reply without a positive prompt token count
    is charged at worst case rather than recorded as free.
    """
    choices = getattr(response, "choices", None) or []
    choice = choices[0] if choices else None
    message = getattr(choice, "message", None)
    usage = getattr(response, "usage", None)
    details = getattr(usage, "completion_tokens_details", None)
    result = LLMResponse(
        text=getattr(message, "content", None) or "",
        finish_reason=getattr(choice, "finish_reason", None) or "",
        prompt_tokens=getattr(usage, "prompt_tokens", None) or 0,
        completion_tokens=getattr(usage, "completion_tokens", None) or 0,
        reasoning_tokens=getattr(details, "reasoning_tokens", None) or 0,
        cost_usd=0.0,
    )
    if result.prompt_tokens > 0:
        result.cost_usd = _cost(price, result.prompt_tokens, result.completion_tokens)
        return result, "pricing"
    result.cost_usd = worst
    return result, "worst_case"


def _write_cache(cache_file: Path, result: LLMResponse):
    """Cache a reply; a failed write only loses the cache entry, never the paid-for reply."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in asdict(result).items() if k not in ("cached", "attempts")}, f)
    except (OSError, ValueError):
        pass
