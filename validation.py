"""
Python-side checks on each proposed edit and on the final article.

These rules are conservative safety heuristics, not proof that a correction
is right. An edit is accepted only when its quoted evidence is found word for
word, backs the new value at the same place in the text (shared anchor words
nearby), and contradicts the old value. Anything that cannot be established
is skipped, never guessed.
"""

import re
import difflib
from article_utils import normalize, find_annotation_leak

MONTHS = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
          'september', 'october', 'november', 'december'}
WEEKDAYS = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
NEGATIONS = {'not', 'no', 'never'}
STOP = set("""
a an and are as at be been being but by can could did do does for from had has have he her
here his how i if in into is it its may might more most much must no nor not of on once only or
other our out over own same she should so some such than that the their them then there these
they this those through to too under until up very was we were what when where which while who
whom why will with would you your also after before about against between during each few
further just new now said says per via amid among within without across along around yet
""".split())
FACT_TYPES = ('NUM', 'DATE', 'NEG', 'ENT')

TOKEN = re.compile(r"\d+(?:[.,/]\d+)*|[^\W\d_]+")
ANCHOR_WINDOW = 8      # context words taken on each side of an edit
PROXIMITY = 10         # max token distance between a new value and an anchor
MAX_GLUE_WORDS = 2     # new words allowed without support (e.g. "unlike", "come")
MAX_DELETE_WORDS = 25  # longer deletions need a linked annotation
MAX_ANNOTATION_NEW_WORDS = MAX_GLUE_WORDS   # annotation-backed edits: new words beyond the corrected values
# annotation corrections are often notes to an editor; these never belong in a published replacement
EDITORIAL_MARKS = ('(', ')', '[', ']', ';', '/', '—', '–')
EDITORIAL_WORDS = {'correct', 'correctly', 'corrected', 'correction', 'incorrect', 'incorrectly', 'actually',
                   'should', 'source', 'sources', 'annotation', 'error', 'erroneous', 'erroneously', 'mistakenly',
                   'wrongly', 'instead', 'note', 'described', 'referred', 'references', 'purportedly', 'clarify'}


def tokens(text: str):
    """Lower-case word and number tokens; commas inside numbers are dropped."""
    return [t.replace(',', '') if t[0].isdigit() else t for t in TOKEN.findall(normalize(text))]


def content_words(text: str):
    return {t for t in tokens(text) if t.isalpha() and len(t) >= 3 and t not in STOP
            and t not in MONTHS and t not in WEEKDAYS}


def _entities(text: str):
    """Runs of capitalized words, as token tuples (e.g. ('kiran', 'c', 'nair'))."""
    found, run = set(), []
    words = text.split()
    for i, word in enumerate(words + ['']):
        bare = word.strip('"\'()[]“”‘’')
        is_cap = bare[:1].isupper()
        is_link = bare in ('of', 'de', 'and') and run and i + 1 < len(words) and words[i + 1][:1].isupper()
        if is_cap or is_link:
            run.append(bare)
            initial = len(bare.rstrip('.')) == 1
            if bare.endswith((',', ';', ':')) or (bare.endswith('.') and not initial):
                found.add(tuple(tokens(' '.join(run))))
                run = []
        else:
            if run:
                found.add(tuple(tokens(' '.join(run))))
            run = []
    entities = set()
    for ent in found:
        ent = tuple(t for t in ent if not t.isdigit())
        while ent and ent[0] in STOP:
            ent = ent[1:]
        if ent and not all(t in STOP or t in MONTHS or t in WEEKDAYS for t in ent):
            entities.add(ent)
    return entities


def facts(text: str):
    """Typed fact values in text: numbers, dates, negations and named entities."""
    toks = tokens(text)
    return {
        'NUM': {t for t in toks if t[0].isdigit()},
        'DATE': {t for t in toks if t in MONTHS or t in WEEKDAYS},
        'NEG': {t for t in toks if t in NEGATIONS},
        'ENT': _entities(text),
    }


def _contains_seq(haystack, needle) -> bool:
    n = len(needle)
    return n > 0 and any(tuple(haystack[i:i + n]) == tuple(needle) for i in range(len(haystack) - n + 1))


def _present(value, kind, ev_tokens, ev_facts) -> bool:
    if kind == 'ENT':
        return _contains_seq(ev_tokens, value)
    return value in ev_facts[kind]


def _positions(value, kind, ev_tokens):
    first = value[0] if kind == 'ENT' else value
    return [i for i, t in enumerate(ev_tokens) if t == first]


def _strip_emphasis(text: str) -> str:
    # markdown emphasis markers (*bold*, _italic_) are formatting, not words a quote has to reproduce
    return re.sub(r'[*_]+', '', text)


def _quote_found(quote: str, text: str) -> bool:
    """
    Every part of quote (split at ellipses) appears in text, in order, ignoring markdown emphasis.
    A quote that stops early may end with a full stop where the source sentence goes on
    ("... (2023)." for "... (2023), a woman ..."); that one closing mark is not required to match.
    """
    if _quote_parts_found(quote, text):
        return True
    trimmed = re.sub(r'[.!?]\s*$', '', quote.strip())
    return trimmed != quote.strip() and _quote_parts_found(trimmed, text)


def _quote_parts_found(quote: str, text: str) -> bool:
    haystack = _strip_emphasis(normalize(text))
    pos = 0
    for part in re.split(r'\.\.\.|…', _strip_emphasis(normalize(quote))):
        part = part.strip(' "\'')
        if not part:
            continue
        found = haystack.find(part, pos)
        if found == -1:
            return False
        pos = found + len(part)
    return True


def linked_annotation(find: str, annotations):
    """The annotation whose quoted error refers to the same text as find, if any."""
    target = normalize(find).strip(' "\'')
    target_words = content_words(find)
    for entry in annotations:
        error = normalize(entry.get('error', '')).strip(' "\'')
        if not error:
            continue
        if error in target or target in error:
            return entry
        shared = target_words & content_words(error)
        if target_words and len(shared) / len(target_words) >= 0.5:
            return entry
    return None


def _anchors(body: str, start: int, end: int, find: str, replace: str):
    """Content words around the edit (and unchanged words inside it) that tie evidence to this spot."""
    line_start = body.rfind('\n', 0, start) + 1
    line_end = body.find('\n', end)
    line_end = len(body) if line_end == -1 else line_end
    before = [t for t in tokens(body[line_start:start]) if t in content_words(t)][-ANCHOR_WINDOW:]
    after = [t for t in tokens(body[end:line_end]) if t in content_words(t)][:ANCHOR_WINDOW]
    unchanged = content_words(find) & content_words(replace)
    return set(before) | set(after) | unchanged


def _aligned(anchors, ev_tokens, values):
    """Enough anchors appear in the evidence, and each new value sits near one of them."""
    anchor_pos = [i for i, t in enumerate(ev_tokens) if t in anchors]
    needed = 2 if len(anchors) >= 2 else 1
    if len({ev_tokens[i] for i in anchor_pos}) < needed:
        return False
    for kind, value in values:
        positions = _positions(value, kind, ev_tokens)
        if not any(abs(p - a) <= PROXIMITY for p in positions for a in anchor_pos):
            return False
    return True


def _source_conflict(removed, anchors, source: str) -> bool:
    """The source states an old value next to the same anchors (so the annotation is contradicted)."""
    src_tokens = tokens(source)
    anchor_pos = [i for i, t in enumerate(src_tokens) if t in anchors]
    for kind, value in removed:
        for p in _positions(value, kind, src_tokens):
            if kind == 'ENT' and tuple(src_tokens[p:p + len(value)]) != tuple(value):
                continue
            near = {src_tokens[a] for a in anchor_pos if abs(a - p) <= PROXIMITY}
            if len(near) >= 2:
                return True
    return False


def _flat(fact_sets):
    return [(kind, value) for kind in FACT_TYPES for value in sorted(fact_sets[kind])]


def _words(text: str):
    return [w.strip('.,:;!?"\'()[]“”‘’') for w in normalize(text).split()]


def _annotation_commentary(find: str, replace: str, introduced):
    """
    Why an annotation-backed replacement is not fit to publish, or None.
    Only the corrected values may be new; anything more is likely the annotation's explanation.
    """
    if any(replace.count(m) > find.count(m) for m in EDITORIAL_MARKS):
        return 'annotation_commentary'
    fact_tokens = {t for k in ('NUM', 'DATE', 'NEG') for t in introduced[k]} | {t for e in introduced['ENT'] for t in e}
    old = set(_words(find))
    new = [w for w in _words(replace) if w and w not in old and not set(tokens(w)) <= fact_tokens]
    if any(t in EDITORIAL_WORDS for w in new for t in tokens(w)):
        return 'annotation_commentary'
    if len(new) > MAX_ANNOTATION_NEW_WORDS:
        return 'annotation_not_minimal'
    return None


def check_edit(edit, start: int, end: int, body: str, source: str, annotations):
    """
    Decide whether a located edit is supported.

    Returns:
        (support, None) with support 'source_supported' or 'annotation_supported',
        or (None, reason) when the edit must be skipped.
    """
    find_facts, repl_facts = facts(edit.find), facts(edit.replace)
    introduced = {k: repl_facts[k] - find_facts[k] for k in FACT_TYPES}
    removed = {k: find_facts[k] - repl_facts[k] for k in FACT_TYPES}
    entity_words = {t for ent in introduced['ENT'] for t in ent}
    new_terms = content_words(edit.replace) - content_words(edit.find) - entity_words
    old_terms = content_words(edit.find) - content_words(edit.replace) - {t for e in removed['ENT'] for t in e}
    anchors = _anchors(body, start, end, edit.find, edit.replace)

    if edit.op == 'delete' or not (any(introduced.values()) or new_terms):
        return _check_deletion(edit, body, start, end, source, annotations, removed, anchors)

    ev_tokens = tokens(edit.evidence)
    if len(ev_tokens) < 3:
        return None, 'evidence_too_short'
    if _quote_found(edit.evidence, source):
        support = 'source_supported'
    else:
        entry = linked_annotation(edit.find, annotations)
        if entry and _quote_found(edit.evidence, entry.get('correction', '')):
            support = 'annotation_supported'
            commentary = _annotation_commentary(edit.find, edit.replace, introduced)
            if commentary:
                return None, commentary
        elif any(_quote_found(edit.evidence, a.get('correction', '')) for a in annotations):
            return None, 'annotation_unlinked'
        else:
            return None, 'evidence_not_found'

    removed_kinds = {k for k in FACT_TYPES if removed[k]}
    if 'DATE' in removed_kinds:
        removed_kinds.add('NUM')   # a date may gain or lose its day number ("June 2025" -> "September 9, 2025")
    if removed_kinds and any(introduced[k] and k not in removed_kinds for k in ('NUM', 'DATE', 'ENT')):
        return None, 'type_mismatch'

    ev_facts = facts(edit.evidence)
    new_values = _flat(introduced)
    if not all(_present(v, k, ev_tokens, ev_facts) for k, v in new_values):
        return None, 'fact_not_in_evidence'
    if introduced['NEG']:
        words = tokens(edit.replace)
        i = next(n for n, t in enumerate(words) if t in introduced['NEG'])
        if not _contains_seq(ev_tokens, words[i:i + 2]):
            return None, 'fact_not_in_evidence'

    ev_words = set(ev_tokens)
    missing = [w for w in new_terms if w not in ev_words]
    if len(missing) > MAX_GLUE_WORDS or (new_terms and not new_values and len(missing) == len(new_terms)):
        return None, 'unsupported_rewrite'

    old_items = _flat(removed) + [('TERM', t) for t in sorted(old_terms)]
    if old_items and all(t in ev_words if k == 'TERM' else _present(t, k, ev_tokens, ev_facts)
                         for k, t in old_items):
        return None, 'no_contradiction'

    supported_terms = [('TERM', w) for w in sorted(new_terms) if w in ev_words]
    if not _aligned(anchors - set(old_terms), ev_tokens, new_values or supported_terms):
        return None, 'evidence_not_slot_aligned'

    if support == 'annotation_supported' and _source_conflict(_flat(removed), anchors, source):
        return None, 'source_conflict'
    return support, None


def _check_deletion(edit, body, start, end, source, annotations, removed, anchors):
    """
    A deletion needs a linked annotation, or source evidence that states a
    different value of the same kind at the same place. A claim's absence from
    the source is never enough.
    """
    line_start = body.rfind('\n', 0, start) + 1
    line_end = body.find('\n', end)
    line = body[line_start:len(body) if line_end == -1 else line_end]
    if edit.find.strip() == line.strip():
        return None, 'delete_whole_line'
    if linked_annotation(edit.find, annotations):
        return 'annotation_supported', None

    removed_words = len(tokens(edit.find)) - len(tokens(edit.replace))
    if removed_words > MAX_DELETE_WORDS:
        return None, 'delete_too_long'
    if not any(removed[k] for k in ('NUM', 'DATE', 'ENT')):
        return None, 'delete_generic'
    if len(tokens(edit.evidence)) < 3 or not _quote_found(edit.evidence, source):
        return None, 'delete_unsupported'

    ev_tokens, ev_facts = tokens(edit.evidence), facts(edit.evidence)
    if not _aligned(anchors, ev_tokens, []):
        return None, 'delete_unsupported'
    # the evidence must state a different value of the same kind, one the article line does not already contain
    line_facts = facts(line)
    contradicted = any(
        any(not _present(v, k, ev_tokens, ev_facts) for v in removed[k]) and (ev_facts[k] - line_facts[k])
        for k in ('NUM', 'DATE', 'ENT')
    )
    if not contradicted:
        return None, 'delete_unsupported'
    return 'source_supported', None


def check_output(body: str, text: str, applied):
    """
    Check the final article.

    Returns:
        (critical, warnings): any critical failure means the output must not be used.
    """
    critical, warnings = [], []
    if not text.strip():
        critical.append('empty_output')
    leak = find_annotation_leak(text)
    if leak:
        critical.append(f'annotation_leak: {leak}')

    old_lines, new_lines = body.split('\n'), text.split('\n')
    if len(old_lines) != len(new_lines):
        critical.append('line_count_changed')
    else:
        for n, (old, new) in enumerate(zip(old_lines, new_lines), 1):
            old_mark, new_mark = re.match(r'\s*#*', old).group(), re.match(r'\s*#*', new).group()
            if old_mark != new_mark:
                critical.append(f'heading_marker_changed: line {n}')
            elif old_mark.strip() and old != new:
                warnings.append(f'heading_text_changed: line {n}')

    ratio = len(text) / max(1, len(body))
    if not 0.85 <= ratio <= 1.10:
        warnings.append(f'length_ratio: {ratio:.2f}')
    old_words, new_words = body.split(), text.split()
    changed = 1 - difflib.SequenceMatcher(None, old_words, new_words, autojunk=False).ratio()
    if changed > 0.15:
        warnings.append(f'changed_words: {changed:.0%}')
    if len(applied) > 15:
        warnings.append(f'many_edits: {len(applied)}')

    out_tokens, out_facts = tokens(text), facts(text)
    for edit in applied:
        old = facts(edit.find)
        new = facts(edit.replace)
        for kind in ('NUM', 'ENT'):
            for value in old[kind] - new[kind]:
                if _present(value, kind, out_tokens, out_facts):
                    shown = ' '.join(value) if kind == 'ENT' else value
                    warnings.append(f'possible_unfixed_duplicate: {shown}')
    return critical, warnings
