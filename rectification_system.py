"""
Article Rectification System

run() takes the AI-generated article and its source, asks the LLM for
targeted edits, keeps only the edits Python can verify, and applies them to
the article body. Anything that goes wrong falls back to the article with its
annotation block removed.
"""

from dataclasses import dataclass, field
import llm_client
from article_utils import split_annotations, parse_annotations
from prompts import build_messages, repair_messages
from edits import parse_edits, locate, resolve_overlaps, apply_edits, EditParseError
from validation import check_edit, check_output

MAX_OUTPUT_TOKENS = 4096


@dataclass
class RectificationResult:
    text: str
    status: str                 # rectified | no_change | no_valid_edits | fallback
    reason: str = ""
    applied: list = field(default_factory=list)
    skipped: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    usage: dict = field(default_factory=dict)


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cost_usd")


def _add_usage(usage: dict, response):
    """Add one request's tokens, recorded cost and provider attempts to usage."""
    if response is None:
        return
    for key in USAGE_KEYS:
        usage[key] = usage.get(key, 0) + (getattr(response, key, 0) or 0)
    usage["calls"] = usage.get("calls", 0) + (getattr(response, "attempts", 0) or 0)


def _request_edits(messages: list, article_id: str, usage: dict):
    """Ask for edits, with one retry if the reply is not usable JSON."""
    response = llm_client.complete(messages, MAX_OUTPUT_TOKENS, article_id)
    _add_usage(usage, response)
    try:
        return parse_edits(response.text)
    except EditParseError as e:
        retry = llm_client.complete(repair_messages(messages, response.text, str(e)), MAX_OUTPUT_TOKENS, article_id)
        _add_usage(usage, retry)
        return parse_edits(retry.text)


def run(ai_generated_content: str, source_content: str, article_id: str = "") -> RectificationResult:
    """
    Rectify an AI-generated article against its source.

    Args:
        ai_generated_content: The AI-generated article, possibly with an annotation block
        source_content: The source article(s) it was generated from

    Returns:
        RectificationResult: the article text and how it was produced
    """
    body, raw_annotations = split_annotations(ai_generated_content)
    annotations = parse_annotations(raw_annotations)
    usage = {}

    def fallback(reason):
        return RectificationResult(text=body, status="fallback", reason=reason, usage=usage)

    try:
        candidates, rejected = _request_edits(build_messages(source_content, body, raw_annotations), article_id, usage)
    except llm_client.LLMError as e:
        _add_usage(usage, e.usage)
        return fallback(e.reason)
    except EditParseError as e:
        return fallback(f"parse_error: {e}")

    skipped = [{"find": str(item.get("find", "")) if isinstance(item, dict) else "", "reason": reason}
               for item, reason in rejected]
    located = []
    for edit in candidates:
        try:
            item, reason = locate(body, edit)
            if reason is None and body[item.start:item.end] == edit.replace:
                reason = "noop"
            if reason is None:
                item.support, reason = check_edit(edit, item.start, item.end, body, source_content, annotations)
        except Exception as e:  # one malformed edit must not discard the others
            reason = f"validation_error: {type(e).__name__}"
        if reason is None:
            located.append(item)
        else:
            skipped.append({"find": edit.find, "reason": reason})

    kept, overlapping = resolve_overlaps(located)
    skipped += [{"find": item.edit.find, "reason": reason} for item, reason in overlapping]

    try:
        text = apply_edits(body, kept)
    except AssertionError as e:
        return fallback(f"critical: {e}")
    critical, warnings = check_output(body, text, [item.edit for item in kept])
    if critical:
        return fallback(f"critical: {'; '.join(critical)}")

    applied = [{"op": i.edit.op, "find": i.edit.find, "replace": i.edit.replace, "support": i.support,
                "reason": i.edit.reason} for i in sorted(kept, key=lambda x: x.start)]
    if applied and text != body:
        status = "rectified"
    else:
        status = "no_valid_edits" if candidates or rejected else "no_change"
    return RectificationResult(text=text, status=status, applied=applied, skipped=skipped,
                               warnings=warnings, usage=usage)
