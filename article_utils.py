"""
Shared helpers for reading AI-generated articles and printing safely.

Standard library only, so these work even when the LLM dependencies are absent.
"""

import re
import sys
import json

# A line that is only an "Error Annotations" heading, in any of the styles seen
# in the dataset: "**Error Annotations:**", "## Error Annotations:", etc. The
# heading may be followed on the same line by the start of the JSON payload.
# It must occupy its own line so a sentence that merely mentions the phrase is
# never treated as a header.
ANNOTATION_HEADER = re.compile(
    r"^[ \t]*(?:\*\*|#{1,6}[ \t]*)?"
    r"Error[ \t]+Annotations"
    r"[ \t]*:?[ \t]*(?:\*\*)?[ \t]*:?"
    r"[ \t]*(?:(?:\[|\{|```).*)?\r?$",
    re.IGNORECASE | re.MULTILINE,
)


def split_annotations(text: str):
    """
    Split an AI-generated article into its body and its annotation block.

    Everything from the first annotation header to the end of the text is
    treated as the annotation block. Its contents are never parsed, so
    malformed JSON cannot break the split.

    Returns:
        (body, annotations): body has trailing whitespace removed and is
        otherwise an exact prefix of the input; annotations is the raw text
        after the header, or None when no header is present.
    """
    match = ANNOTATION_HEADER.search(text)
    if not match:
        return text.rstrip(), None
    return text[:match.start()].rstrip(), text[match.end():]


def strip_annotations(text: str) -> str:
    """Return the article body with any annotation block removed."""
    return split_annotations(text)[0]


# Text that must never appear in a saved article: annotation headers, JSON
# annotation keys and code fences.
LEAK_MARKERS = ('"location":', '"error":', '"correction":', '"error_type":', '```')


def find_annotation_leak(text: str):
    """Return the first annotation artifact found in text, or None."""
    match = ANNOTATION_HEADER.search(text)
    if match:
        return match.group(0).strip()
    return next((marker for marker in LEAK_MARKERS if marker in text), None)


_ANNOTATION_FIELD = re.compile(r'^\s*"(error|correction)"\s*:\s*(.*?)\s*,?\s*$', re.MULTILINE)


def parse_annotations(raw):
    """
    Extract {error, correction} pairs from an annotation block.

    Uses JSON when the block is valid and falls back to reading the fields
    line by line, so malformed blocks still yield usable hints.
    """
    if not raw:
        return []
    text = raw.strip().strip('`').strip()
    try:
        data = json.loads(text)
        return [
            {'error': str(item.get('error', '')), 'correction': str(item.get('correction', ''))}
            for item in data if isinstance(item, dict)
        ]
    except (ValueError, TypeError, AttributeError):
        pass

    entries, current = [], {}
    for field, value in _ANNOTATION_FIELD.findall(text):
        try:
            value = str(json.loads(value))
        except ValueError:
            value = value.strip('"\\').replace('\\"', '"')
        if field == 'error' and current:
            entries.append(current)
            current = {}
        current[field] = value
    if current:
        entries.append(current)
    return [{'error': e.get('error', ''), 'correction': e.get('correction', '')} for e in entries]


_CHAR_MAP = str.maketrans({
    '‘': "'", '’': "'", '“': '"', '”': '"',
    '–': '-', '—': '-', ' ': ' ',
})


def normalize_with_map(text: str):
    """
    Normalize quotes, dashes and whitespace runs for matching.

    Returns:
        (normalized, index_map): index_map[i] is the position in text of
        normalized[i], so a match can be mapped back to the original.
    """
    chars, index_map = [], []
    translated = text.translate(_CHAR_MAP)
    for i, ch in enumerate(translated):
        if ch.isspace():
            if chars and chars[-1] == ' ':
                continue
            ch = ' '
        chars.append(ch)
        index_map.append(i)
    return ''.join(chars), index_map


def normalize(text: str) -> str:
    """Case-insensitive normalized form used for comparing text."""
    return normalize_with_map(text)[0].casefold()


def configure_console():
    """
    Stop print() from crashing on characters the console encoding cannot
    represent (e.g. cp1252 on a redirected Windows stdout).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
