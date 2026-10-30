"""
The single place that calls the LLM provider.

Every attempt is checked against the spending caps first and recorded in a
local ledger afterwards. Groq has no balance endpoint (budget_checker.py only
works with a LiteLLM proxy), so the ledger is the record of spend.
"""

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
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

REASONING_EFFORT = "medium"
TIMEOUT_SECONDS = 120
MAX_ATTEMPTS = 3               # first try plus two retries, for 429 / timeout / 5xx only
MAX_BACKOFF_SECONDS = 30

_sleep = time.sleep
_disabled_reason = None        # set after a non-recoverable error; stops further calls


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


def complete(messages: list, max_tokens: int, article_id: str = "") -> LLMResponse:
    """
    Send one chat request and return the reply.

    Raises:
        LLMError: with reason 'disabled', 'missing_config', 'pricing_unavailable',
        'budget_exceeded', 'truncated', or 'llm_error:<type>'.
    """
    global _disabled_reason
    if _disabled_reason:
        raise LLMError("disabled", _disabled_reason)

    model, api_key, api_base = os.getenv("LLM_MODEL_NAME"), os.getenv("LLM_API_KEY"), os.getenv("LLM_API_BASE")
    if not model or not api_key:
        _disabled_reason = "LLM_MODEL_NAME or LLM_API_KEY not set"
        raise LLMError("missing_config", _disabled_reason)
    price = PRICING.get(model)
    if price is None:
        _disabled_reason = f"no verified pricing for model {model}"
        raise LLMError("pricing_unavailable", _disabled_reason)

    params = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
              "reasoning_effort": REASONING_EFFORT, "response_format": {"type": "json_object"}}
    cache_file = _cache_path(params)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        return LLMResponse(**{**cached, "cost_usd": 0.0, "cached": True, "attempts": 0})

    cap = spend_cap()
    charged, sent = 0.0, 0

    def failure(reason, message, reply=None):
        usage = reply or LLMResponse("", "", 0, 0, 0, 0.0)
        usage.cost_usd, usage.attempts = charged, sent
        return LLMError(reason, message, usage)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        worst = _worst_case(price, messages, max_tokens)
        total = spent()
        if total + worst > cap or total + worst > HARD_CAP_USD:
            _disabled_reason = f"spent ${total:.4f}, next call could cost ${worst:.4f}, cap ${cap:.2f}"
            raise failure("budget_exceeded", _disabled_reason)

        entry = {"time": datetime.now(timezone.utc).isoformat(), "article_id": article_id,
                 "model": model, "attempt": attempt, "pricing_source": PRICING_SOURCE}
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
