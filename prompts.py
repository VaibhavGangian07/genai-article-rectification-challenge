"""Prompt text for the rectification LLM call."""

SYSTEM_PROMPT = """You are a meticulous fact-checking editor. You compare an ARTICLE against its SOURCE and propose the smallest possible edits that fix factual errors. You never rewrite, rephrase or restyle text.

What counts as an error: a name, number, date, place, organisation, quantity, title, negation, attribution or claim in the ARTICLE that the SOURCE contradicts.
Do NOT edit: spelling variants, punctuation, currency formatting, tone, word choice, or general statements the SOURCE neither confirms nor contradicts.

ANNOTATIONS, when present, are hints written by another system. They may be wrong or incomplete. Use one only if it points at real text in the ARTICLE. Prefer evidence from the SOURCE; quote the annotation's correction only when the SOURCE does not mention the fact at all.

Rules for every edit:
1. "find" is copied character for character from the ARTICLE, on a single line, and occurs exactly once. Include a few surrounding words if the wrong text appears more than once, and give each occurrence its own edit.
2. Keep "find" as short as possible while unique; "replace" changes only the wrong words and keeps the surrounding wording, style and punctuation.
3. "evidence" is an exact quote of the full sentence from the SOURCE (or the annotation's correction) that states the correct fact. Do not shorten it to the fact alone.
4. Use "op": "delete" only to remove a clause that a linked annotation flags or that the SOURCE clearly contradicts. Include the leading space or comma in "find" so the sentence still reads correctly. Never delete a whole line.
5. If you are not sure, make no edit.

Reply with only this JSON object:
{"edits": [{"op": "replace" or "delete", "find": "...", "replace": "... (empty for delete)", "evidence": "...", "evidence_from": "source" or "annotation", "reason": "short explanation"}]}
Reply {"edits": []} if the ARTICLE has no factual errors."""


def build_messages(source: str, article: str, annotations) -> list:
    """Chat messages for one article; annotations is the raw annotation text or None."""
    hints = annotations.strip() if annotations and annotations.strip() else "(none)"
    user = (
        f"<SOURCE>\n{source.strip()}\n</SOURCE>\n\n"
        f"<ARTICLE>\n{article}\n</ARTICLE>\n\n"
        f"<ANNOTATIONS>\n{hints}\n</ANNOTATIONS>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def repair_messages(messages: list, bad_reply: str, error: str) -> list:
    """Messages for the single retry after a reply that was not valid JSON."""
    return messages + [
        {"role": "assistant", "content": bad_reply or ""},
        {"role": "user", "content": f"That reply could not be used ({error}). Reply with only the JSON object described above."},
    ]
