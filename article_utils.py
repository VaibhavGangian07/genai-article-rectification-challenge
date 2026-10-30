"""
Shared helpers for reading AI-generated articles and printing safely.

Standard library only, so these work even when the LLM dependencies are absent.
"""

import re
import sys

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
