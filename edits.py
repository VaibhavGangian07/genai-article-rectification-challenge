"""
Targeted edits: parse the LLM's JSON, locate each edit in the article and
apply the accepted ones deterministically.

Nothing here decides whether an edit is factually justified (see
validation.py); this module only guarantees that edits land exactly where
intended and that untouched text is left byte-for-byte unchanged.
"""

import re
import json
from dataclasses import dataclass
from article_utils import normalize_with_map

MAX_FIND_CHARS = 300
FORBIDDEN_IN_REPLACE = ('```', 'error annotations')


class EditParseError(ValueError):
    """The LLM response is not a JSON object with an "edits" list."""


@dataclass
class Edit:
    op: str
    find: str
    replace: str
    evidence: str
    evidence_from: str
    reason: str


@dataclass
class LocatedEdit:
    edit: Edit
    start: int
    end: int
    support: str = ""


def parse_edits(raw: str):
    """
    Parse the LLM response into edits.

    Returns:
        (edits, rejected): valid Edit objects, and (item, reason) pairs for
        entries that do not match the schema.
    """
    text = (raw or '').strip()
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text)
    start, end = text.find('{'), text.rfind('}')
    if start == -1 or end < start:
        raise EditParseError("no JSON object in response")
    try:
        data = json.loads(text[start:end + 1])
    except ValueError as e:
        raise EditParseError(f"invalid JSON: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get('edits'), list):
        raise EditParseError('missing "edits" list')

    edits, rejected = [], []
    for item in data['edits']:
        edit, reason = _to_edit(item)
        if edit:
            edits.append(edit)
        else:
            rejected.append((item, reason))
    return edits, rejected


def _to_edit(item):
    if not isinstance(item, dict):
        return None, 'schema_invalid: not an object'
    op = item.get('op')
    find = item.get('find')
    replace = item.get('replace') or ''
    evidence = item.get('evidence') or ''
    if op not in ('replace', 'delete'):
        return None, 'schema_invalid: op'
    if not isinstance(find, str) or not find.strip() or len(find) > MAX_FIND_CHARS:
        return None, 'schema_invalid: find'
    if not isinstance(replace, str) or not isinstance(evidence, str):
        return None, 'schema_invalid: types'
    if '\n' in find or '\n' in replace:
        return None, 'multiline_not_allowed'
    if op == 'replace':
        if not replace.strip() or replace == find:
            return None, 'schema_invalid: replace'
        if replace.lstrip().startswith('#') or any(f in replace.lower() for f in FORBIDDEN_IN_REPLACE):
            return None, 'forbidden_markup'
    elif replace:
        return None, 'schema_invalid: delete with replacement'
    return Edit(op, find, replace, evidence, str(item.get('evidence_from') or ''),
                str(item.get('reason') or '')), None


def _occurrences(text: str, needle: str):
    """Start positions of needle in text (overlapping), on word boundaries."""
    positions, i = [], text.find(needle)
    while i != -1:
        end = i + len(needle)
        left_ok = not (needle[0].isalnum() and i > 0 and text[i - 1].isalnum())
        right_ok = not (needle[-1].isalnum() and end < len(text) and text[end].isalnum())
        if left_ok and right_ok:
            positions.append(i)
        i = text.find(needle, i + 1)
    return positions


def locate(body: str, edit: Edit):
    """
    Find the single span an edit refers to.

    Returns:
        (LocatedEdit, None) on a unique match, or (None, reason).
    """
    hits = _occurrences(body, edit.find)
    if len(hits) == 1:
        return LocatedEdit(edit, hits[0], hits[0] + len(edit.find)), None
    if len(hits) > 1:
        return None, f'ambiguous({len(hits)})'

    norm_body, index_map = normalize_with_map(body)
    norm_find = normalize_with_map(edit.find)[0]
    hits = _occurrences(norm_body, norm_find) if norm_find.strip() else []
    if len(hits) == 1:
        start = index_map[hits[0]]
        end = index_map[hits[0] + len(norm_find) - 1] + 1
        return LocatedEdit(edit, start, end), None
    if len(hits) > 1:
        return None, f'ambiguous({len(hits)})'
    return None, 'not_found'


def resolve_overlaps(located):
    """
    Drop exact duplicates and reject every edit that overlaps another.

    Returns:
        (kept, rejected): rejected holds (LocatedEdit, reason) pairs.
    """
    unique = {}
    for item in located:
        unique.setdefault((item.start, item.end, item.edit.replace), item)
    items = sorted(unique.values(), key=lambda x: (x.start, x.end))
    overlapping = set()
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if b.start >= a.end:
                break
            overlapping.update((id(a), id(b)))
    kept = [x for x in items if id(x) not in overlapping]
    rejected = [(x, 'overlap') for x in items if id(x) in overlapping]
    return kept, rejected


def apply_edits(body: str, located) -> str:
    """
    Apply non-overlapping located edits to body, right to left.

    A deletion removes only its span, then tidies the join: a doubled space
    becomes one, and a space left before punctuation is dropped.
    """
    text = body
    for item in sorted(located, key=lambda x: x.start, reverse=True):
        before, after = text[:item.start], text[item.end:]
        replacement = item.edit.replace
        if not replacement:
            if before.endswith(' ') and (after.startswith(' ') or after[:1] in tuple(',.;:)')):
                before = before[:-1]
        text = before + replacement + after
    _check_untouched_lines(body, text, located)
    return text


def _check_untouched_lines(body: str, text: str, located):
    """Every line no edit touched must be unchanged, and no line may be added or removed."""
    old_lines, new_lines = body.split('\n'), text.split('\n')
    if len(old_lines) != len(new_lines):
        raise AssertionError("edit changed the number of lines")
    touched, offset = set(), 0
    starts = []
    for line in old_lines:
        starts.append(offset)
        offset += len(line) + 1
    for item in located:
        for n, line_start in enumerate(starts):
            line_end = line_start + len(old_lines[n])
            if item.start <= line_end and item.end >= line_start:
                touched.add(n)
    for n, (old, new) in enumerate(zip(old_lines, new_lines)):
        if n not in touched and old != new:
            raise AssertionError(f"untouched line {n + 1} changed")
