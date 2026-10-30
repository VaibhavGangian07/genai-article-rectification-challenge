"""
Evaluation harness: score article texts against the human-rectified references.

Modes
  baseline     the AI article with its annotation block removed (no corrections)
  gold_replay  edits derived from the human references, sent through the real
               engine (locate -> validate -> apply -> output checks) in place of
               an LLM reply. This tests the validators and edit application only;
               it says nothing about whether an LLM can find these corrections.
  candidate    rectified files from --candidate-dir (e.g. a future LLM run)

No provider is ever called: gold replay replaces llm_client.complete.

Usage:
    python evaluate.py
    python evaluate.py --candidate-dir eval_outputs/some_run --label some_run
"""

import re
import json
import difflib
import argparse
from pathlib import Path
from unittest import mock

import llm_client
import rectification_system
from edits import _occurrences
from evaluate_baseline import ROOT, REFERENCE_DIR, read_text, verify_references, compare
from article_utils import split_annotations, parse_annotations, find_annotation_leak, configure_console
from validation import facts, content_words, tokens

SPLIT = {
    'tuning': ['article_001', 'article_002', 'article_005', 'article_006', 'article_008', 'article_009', 'article_010'],
    'holdout': ['article_003', 'article_004', 'article_007'],
}
OUTPUT_DIR = ROOT / 'eval_outputs'
MERGE_GAP = 2           # diff hunks separated by at most this many unchanged words form one region
MAX_CONTEXT = 6         # unchanged words added on each side to make a gold edit's "find" unique
PROTECTED = (ROOT / 'rectified_articles', REFERENCE_DIR)


# ---------------------------------------------------------------- regions

def _words(text: str):
    """Whitespace-separated words with their character spans."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r'\S+', text)]


def diff_regions(old_words, new_words):
    """
    Changed regions between two word lists, as (i1, i2, j1, j2) word ranges.
    Hunks separated by MERGE_GAP or fewer unchanged words are merged, so one
    correction that the diff splits in two counts once.
    """
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    regions = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            continue
        if regions and i1 - regions[-1][1] <= MERGE_GAP and j1 - regions[-1][3] <= MERGE_GAP:
            regions[-1] = (regions[-1][0], i2, regions[-1][2], j2)
        else:
            regions.append((i1, i2, j1, j2))
    return regions


def _position_map(old_words, new_words):
    """Map each word boundary in old to the matching boundary in new (outside changed blocks)."""
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    positions = {}
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'equal':
            for k in range(i2 - i1 + 1):
                positions[i1 + k] = j1 + k
        else:
            positions.setdefault(i1, j1)
            positions[i2] = j2
    return positions


def _overlap(a, b) -> bool:
    """Word ranges overlap; an insertion (empty range) overlaps what touches its position."""
    (s1, e1), (s2, e2) = a, b
    if s1 == e1 or s2 == e2:
        return s1 <= e2 and s2 <= e1
    return s1 < e2 and s2 < e1


def score(base: str, candidate: str, reference: str) -> dict:
    """
    Compare a candidate with the human reference, both measured as changes to
    base (the annotation-stripped AI article).
    """
    b, c, r = [w for w, _, _ in _words(base)], [w for w, _, _ in _words(candidate)], [w for w, _, _ in _words(reference)]
    gold, pred = diff_regions(b, r), diff_regions(b, c)
    to_ref, to_cand = _position_map(b, r), _position_map(b, c)

    # group overlapping gold and predicted regions into components (on base word ranges)
    items = [('gold', g) for g in gold] + [('pred', p) for p in pred]
    parent = list(range(len(items)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for x in range(len(items)):
        for y in range(x + 1, len(items)):
            if _overlap(items[x][1][:2], items[y][1][:2]):
                parent[find(x)] = find(y)
    components = {}
    for i, item in enumerate(items):
        components.setdefault(find(i), []).append(item)

    correct_gold = correct_pred = located_gold = located_pred = 0
    unintended_regions = unintended_words = 0
    for members in components.values():
        g = [m for kind, m in members if kind == 'gold']
        p = [m for kind, m in members if kind == 'pred']
        if g and p:
            lo, hi = min(m[0] for _, m in members), max(m[1] for _, m in members)
            located_gold, located_pred = located_gold + len(g), located_pred + len(p)
            if c[to_cand[lo]:to_cand[hi]] == r[to_ref[lo]:to_ref[hi]]:
                correct_gold, correct_pred = correct_gold + len(g), correct_pred + len(p)
        elif p:
            unintended_regions += len(p)
            unintended_words += sum(max(i2 - i1, j2 - j1) for i1, i2, j1, j2 in p)

    def ratio(n, d):
        return n / d if d else None
    precision, recall = ratio(correct_pred, len(pred)), ratio(correct_gold, len(gold))
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if gold or pred else None)
    return {
        'exact_match': candidate.rstrip() == reference.rstrip(),
        'exact_words': c == r,
        'similarity': compare(candidate, reference)['similarity'],
        'gold_regions': len(gold), 'pred_regions': len(pred),
        'correct_gold': correct_gold, 'correct_pred': correct_pred,
        'located_gold': located_gold, 'located_pred': located_pred,
        'precision': precision, 'recall': recall, 'f1': f1,
        'unintended_regions': unintended_regions, 'unintended_words': unintended_words,
        'annotation_leak': find_annotation_leak(candidate),
        'words': word_changes(b, c),
        'reference_words': word_changes(b, r),
    }


def word_changes(old_words, new_words) -> dict:
    """Words added, removed and modified (replaced one for one) going from old to new."""
    counts = {'added': 0, 'removed': 0, 'modified': 0}
    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        n_old, n_new = i2 - i1, j2 - j1
        if op == 'replace':
            counts['modified'] += min(n_old, n_new)
            counts['added'] += max(0, n_new - n_old)
            counts['removed'] += max(0, n_old - n_new)
        elif op == 'insert':
            counts['added'] += n_new
        elif op == 'delete':
            counts['removed'] += n_old
    return counts


# ---------------------------------------------------------------- gold replay

_SENTENCE_END = re.compile(
    # . ! or ? then whitespace and a capital, digit or quote; not after an initial ("C.") or a title ("Dr.")
    r'(?<![\s(][A-Z]\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bDr\.)(?<!\bSt\.)(?<!\bMrs\.)'
    r'(?<=[.!?])["”\']?\s+(?=["“\'(]?[A-Z0-9])|\n+'
)


def _sentences(text: str):
    return [s.strip() for s in _SENTENCE_END.split(text) if s and len(s.split()) >= 3]


def pick_evidence(find: str, replace: str, line: str, source: str, annotations):
    """
    Heuristically choose the sentence that best supports a gold edit, as an
    LLM following the prompt should: the one containing most of the new values,
    then sharing most words with the article line, preferring the source to an
    annotation's correction.
    """
    new_values = {v for kind in ('NUM', 'DATE', 'NEG') for v in facts(replace)[kind] - facts(find)[kind]}
    new_values |= {t for ent in facts(replace)['ENT'] - facts(find)['ENT'] for t in ent}
    new_values |= content_words(replace) - content_words(find)
    context = (content_words(line) | content_words(replace)) - (content_words(find) - content_words(replace))
    options = [(s, 'source') for s in _sentences(source)]
    options += [(a['correction'].strip().strip('"\''), 'annotation') for a in annotations if a.get('correction')]
    best, best_score = ('', 'source'), None
    for text, origin in options:
        toks = set(tokens(text))
        candidate_score = (len(new_values & toks), len(context & toks), origin == 'source')
        if best_score is None or candidate_score > best_score:
            best, best_score = (text, origin), candidate_score
    return best


def gold_edits(body: str, reference: str, source: str, annotations):
    """
    Turn each changed region between body and reference into an edit in the
    engine's JSON schema. Unchanged words are added on both sides until the
    "find" text is unique, without crossing a line or a neighbouring region.

    Returns:
        (edits, not_derivable): edit dicts, and regions that could not be expressed
    """
    a, b = _words(body), _words(reference)
    line = [body.count('\n', 0, start) for _, start, _ in a]
    regions = diff_regions([w for w, _, _ in a], [w for w, _, _ in b])
    edits, not_derivable = [], []
    for n, (i1, i2, j1, j2) in enumerate(regions):
        changed = ' '.join(w for w, _, _ in a[i1:i2]) + ' -> ' + ' '.join(w for w, _, _ in b[j1:j2])
        # unchanged words available on each side: same line, not part of a neighbouring region
        anchor_line = line[i1] if i1 < len(a) else line[-1]
        left_limit = regions[n - 1][1] if n else 0
        right_limit = regions[n + 1][0] if n + 1 < len(regions) else len(a)
        left = 0
        while i1 - left - 1 >= left_limit and line[i1 - left - 1] == (line[i1] if i1 < i2 else anchor_line):
            left += 1
        right = 0
        while i2 + right < right_limit and line[i2 + right] == (line[i2 - 1] if i1 < i2 else anchor_line):
            right += 1
        find = replace = ''
        for k in range(MAX_CONTEXT + 1):
            lo, hi = i1 - min(k, left), i2 + min(k, right)
            if lo == hi:
                continue
            jlo, jhi = j1 - (i1 - lo), j2 + (hi - i2)
            find = body[a[lo][1]:a[hi - 1][2]]
            replace = reference[b[jlo][1]:b[jhi - 1][2]] if jhi > jlo else ''
            if len(_occurrences(body, find)) == 1 or k >= max(left, right):
                break
        if not find or '\n' in find or '\n' in replace:
            not_derivable.append({'region': changed, 'reason': 'crosses a line break or has no context on its line'})
            continue
        line_start = body.rfind('\n', 0, a[i1 if i1 < len(a) else -1][1]) + 1
        line_end = body.find('\n', line_start)
        evidence, origin = pick_evidence(find, replace, body[line_start:line_end if line_end != -1 else len(body)],
                                         source, annotations)
        edits.append({'op': 'replace' if replace else 'delete', 'find': find, 'replace': replace,
                      'evidence': evidence, 'evidence_from': origin, 'reason': f'gold: {changed}'})
    return edits, not_derivable


def replay(ai_text: str, source: str, reference: str, article_id: str):
    """Run the engine with gold edits standing in for the LLM reply."""
    body, raw = split_annotations(ai_text)
    edits, not_derivable = gold_edits(body, reference, source, parse_annotations(raw))
    reply = llm_client.LLMResponse(json.dumps({'edits': edits}), 'stop', 0, 0, 0, 0.0, attempts=0)
    with mock.patch.object(llm_client, 'complete', return_value=reply), \
         mock.patch.object(llm_client, '_completion', side_effect=AssertionError('no provider calls in evaluation')):
        result = rectification_system.run(ai_text, source, article_id)
    return result, edits, not_derivable


# ---------------------------------------------------------------- driver

def _load(article_id: str, mapping: dict):
    entry = mapping[article_id]
    return (read_text(ROOT / entry['ai_generated_file']), read_text(ROOT / entry['source_file']),
            read_text(REFERENCE_DIR / f'{article_id}.txt'))


def evaluate(candidate_dir: Path = None, label: str = 'candidate', output_dir: Path = OUTPUT_DIR):
    """Score every reference article in each mode and write the reports."""
    output_dir = Path(output_dir).resolve()
    if any(output_dir == p or output_dir.is_relative_to(p) for p in PROTECTED):
        raise ValueError(f'Refusing to write evaluation output into {output_dir}')
    verify_references()
    with open(ROOT / 'article_mapping.json', 'r') as f:
        mapping = {a['article_id']: a for a in json.load(f)}

    articles = []
    for split, ids in SPLIT.items():
        for article_id in ids:
            ai, source, reference = _load(article_id, mapping)
            body, raw = split_annotations(ai)
            row = {'article_id': article_id, 'split': split, 'annotated': raw is not None, 'modes': {}}
            row['modes']['baseline'] = score(body, body, reference)

            result, edits, not_derivable = replay(ai, source, reference, article_id)
            replay_row = score(body, result.text, reference)
            replay_row.update(status=result.status, reason=result.reason, gold_edits=len(edits),
                              accepted=len(result.applied), rejected=result.skipped, not_derivable=not_derivable,
                              accepted_support=[a['support'] for a in result.applied], warnings=result.warnings)
            row['modes']['gold_replay'] = replay_row
            (output_dir / 'gold_replay').mkdir(parents=True, exist_ok=True)
            (output_dir / 'gold_replay' / f'{article_id}.txt').write_text(result.text, encoding='utf-8')

            if candidate_dir:
                path = Path(candidate_dir) / f'{article_id}.txt'
                row['modes'][label] = score(body, read_text(path), reference) if path.exists() else {'missing': True}
            articles.append(row)

    summary = summarize(articles)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'article_results.json').write_text(json.dumps(articles, indent=2), encoding='utf-8')
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output_dir / 'evaluation_report.md').write_text(render_report(summary, articles), encoding='utf-8')
    return summary, articles


def summarize(articles):
    """Totals per mode for each group: all, tuning, holdout, annotated, unannotated."""
    groups = {
        'all': articles,
        'tuning': [a for a in articles if a['split'] == 'tuning'],
        'holdout': [a for a in articles if a['split'] == 'holdout'],
        'annotated': [a for a in articles if a['annotated']],
        'unannotated': [a for a in articles if not a['annotated']],
    }
    modes = list(dict.fromkeys(m for a in articles for m in a['modes']))
    summary = {}
    for mode in modes:
        summary[mode] = {}
        for group, rows in groups.items():
            scored = [a['modes'][mode] for a in rows if mode in a['modes'] and not a['modes'][mode].get('missing')]
            total = lambda key: sum(s[key] for s in scored)
            gold, pred = total('gold_regions'), total('pred_regions')
            precision = total('correct_pred') / pred if pred else None
            recall = total('correct_gold') / gold if gold else None
            f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if gold or pred else None)
            words = {k: sum(s['words'][k] for s in scored) for k in ('added', 'removed', 'modified')}
            entry = {
                'articles': len(scored),
                'exact_match': total('exact_match'),
                'exact_words': total('exact_words'),
                'mean_similarity': sum(s['similarity'] for s in scored) / len(scored) if scored else None,
                'gold_regions': gold, 'pred_regions': pred,
                'precision': precision, 'recall': recall, 'f1': f1,
                'located_recall': total('located_gold') / gold if gold else None,
                'unintended_regions': total('unintended_regions'), 'unintended_words': total('unintended_words'),
                'annotation_leaks': sum(1 for s in scored if s['annotation_leak']),
                'words': words,
            }
            if mode == 'gold_replay':
                entry['gold_edits'] = total('gold_edits')
                entry['accepted'] = total('accepted')
                entry['not_derivable'] = sum(len(s['not_derivable']) for s in scored)
                entry['acceptance_rate'] = entry['accepted'] / entry['gold_edits'] if entry['gold_edits'] else None
                reasons = {}
                for s in scored:
                    for r in s['rejected']:
                        key = r['reason'].split('(')[0]
                        reasons[key] = reasons.get(key, 0) + 1
                entry['rejections_by_reason'] = dict(sorted(reasons.items(), key=lambda kv: -kv[1]))
                entry['support'] = {k: sum(s['accepted_support'].count(k) for s in scored)
                                    for k in ('source_supported', 'annotation_supported')}
            summary[mode][group] = entry
    return summary


# ---------------------------------------------------------------- report

def _fmt(value, pct=False):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.0%}' if pct else f'{value:.3f}'
    return str(value)


METRICS_TEXT = """## What the metrics measure

All change-based metrics compare two edits of the same text: the human editor's (AI article -> reference) and the candidate's (AI article -> candidate). Changes are measured on whitespace-separated words; diff hunks separated by at most 2 unchanged words are merged into one **region**.

| Metric | Measures | Limitations |
|---|---|---|
| Exact match | Candidate equals the reference (ignoring trailing whitespace) | All-or-nothing; one stylistic difference fails it. Some references contain stray double spaces |
| Exact words | Same words in the same order, ignoring spacing | Hides whitespace differences that a strict comparison would count |
| Text similarity | Word-level `difflib` ratio, candidate vs reference | **Not a factual-accuracy score.** The uncorrected baseline already scores ~0.95, because errors are a few words in a long text |
| Precision | Share of the candidate's changed regions whose words match the reference exactly | Counts a region as wrong if the wording differs even when the fact is right |
| Recall | Share of the human's changed regions reproduced exactly | Assumes the reference is the only correct fix; includes the human's stylistic rewording, which our engine deliberately does not attempt |
| F1 | Harmonic mean of precision and recall | Inherits both limitations |
| Located recall | Share of human regions the candidate touched at all (right place, any wording) | Says nothing about whether the new text is right |
| Unintended changes | Candidate regions (and words) that overlap no human change | Some may be legitimate fixes the human missed; most indicate collateral damage |
| Annotation leakage | Candidate contains an annotation header, JSON annotation keys or a code fence | Pattern-based; detects the known annotation format |
| Words added / removed / modified | Size of the candidate's change relative to the AI article | Size only; says nothing about correctness |
"""


def render_report(summary, articles) -> str:
    lines = ['# Evaluation report', '',
             'Scored against the 10 human-rectified references in `reference_rectified/`. No provider was called.', '',
             '- **baseline**: the AI article with its annotation block removed, no corrections.',
             '- **gold_replay**: edits derived from the human references, sent through the real engine '
             '(locate, validate, apply, output checks) in place of an LLM reply. This measures the validators and '
             'edit application only. **It is not LLM performance**: the LLM still has to discover these corrections.',
             '- Evidence for gold edits is picked automatically (best-matching source sentence or annotation correction), '
             'so some rejections reflect the evidence picker rather than the validator.', '',
             f"Split: tuning = {', '.join(a[-3:] for a in SPLIT['tuning'])}; holdout = {', '.join(a[-3:] for a in SPLIT['holdout'])}. "
             'Holdout rows are reported, not tuned against.', '']
    lines.append('## Summary')
    lines.append('')
    lines.append('| Mode | Group | Articles | Exact | Exact words | Similarity | Precision | Recall | F1 | Located recall | Unintended regions (words) | Leaks |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for mode, groups in summary.items():
        for group, s in groups.items():
            if not s['articles']:
                continue
            lines.append(f"| {mode} | {group} | {s['articles']} | {s['exact_match']} | {s['exact_words']} | {_fmt(s['mean_similarity'])} | "
                         f"{_fmt(s['precision'], True)} | {_fmt(s['recall'], True)} | {_fmt(s['f1'], True)} | "
                         f"{_fmt(s['located_recall'], True)} | {s['unintended_regions']} ({s['unintended_words']}) | "
                         f"{s['annotation_leaks']} |")
    lines.append('')

    if 'gold_replay' in summary:
        g, base = summary['gold_replay']['all'], summary['baseline']['all']
        lines += ['## Baseline vs gold replay', '',
                  f"- Exact matches: {base['exact_match']} -> {g['exact_match']} of {g['articles']} "
                  f"(same words, ignoring spacing: {base['exact_words']} -> {g['exact_words']})",
                  f"- Recall: {_fmt(base['recall'], True)} -> {_fmt(g['recall'], True)}; "
                  f"precision {_fmt(base['precision'], True)} -> {_fmt(g['precision'], True)}",
                  f"- Mean similarity: {_fmt(base['mean_similarity'])} -> {_fmt(g['mean_similarity'])} "
                  '(small, because similarity hardly moves with factual fixes)',
                  f"- Unintended changes: {base['unintended_regions']} -> {g['unintended_regions']}", '',
                  '## Validator acceptance of gold edits (gold replay)', '',
                  f"{g['accepted']} of {g['gold_edits']} gold edits accepted ({_fmt(g['acceptance_rate'], True)}); "
                  f"{g['not_derivable']} human changes could not be expressed as single-line edits.",
                  f"Accepted by support: {g['support']['source_supported']} source-supported, "
                  f"{g['support']['annotation_supported']} annotation-supported.", '',
                  '| Rejection reason | Count |', '|---|---|']
        lines += [f'| {reason} | {count} |' for reason, count in g['rejections_by_reason'].items()]
        lines.append('')

    lines += ['## Per article', '',
              '| Article | Split | Annotated | Mode | Status | Exact | Similarity | P | R | Unintended | Leak | Words +/-/~ |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for a in articles:
        for mode, s in a['modes'].items():
            if s.get('missing'):
                lines.append(f"| {a['article_id'][-3:]} | {a['split']} | {'yes' if a['annotated'] else 'no'} | {mode} | missing | | | | | | | |")
                continue
            w = s['words']
            lines.append(f"| {a['article_id'][-3:]} | {a['split']} | {'yes' if a['annotated'] else 'no'} | {mode} | "
                         f"{s.get('status', '-')} | {'yes' if s['exact_match'] else 'no'} | {_fmt(s['similarity'])} | "
                         f"{_fmt(s['precision'], True)} | {_fmt(s['recall'], True)} | {s['unintended_regions']} | "
                         f"{'yes' if s['annotation_leak'] else 'no'} | {w['added']}/{w['removed']}/{w['modified']} |")
    lines.append('')

    lines += ['## Rejected and non-derivable gold edits', '']
    for a in articles:
        g = a['modes'].get('gold_replay')
        if g and (g['rejected'] or g['not_derivable']):
            lines.append(f"**{a['article_id']}** ({a['split']})")
            lines += [f"- `{r['reason']}`: {r['find']!r}" for r in g['rejected']]
            lines += [f"- `not derivable ({r['reason']})`: {r['region']!r}" for r in g['not_derivable']]
            lines.append('')
    lines.append(METRICS_TEXT)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate rectification against the human references (no API calls).')
    parser.add_argument('--candidate-dir', type=Path, help='Folder of rectified article_XXX.txt files to score')
    parser.add_argument('--label', default='candidate', help='Name for the candidate mode in the reports')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR, help='Where to write the reports')
    args = parser.parse_args()
    summary, _ = evaluate(args.candidate_dir, args.label, args.output_dir)
    for mode, groups in summary.items():
        s = groups['all']
        print(f"{mode:12} exact={s['exact_match']}/{s['articles']} similarity={_fmt(s['mean_similarity'])} "
              f"P={_fmt(s['precision'], True)} R={_fmt(s['recall'], True)} F1={_fmt(s['f1'], True)} "
              f"unintended={s['unintended_regions']} leaks={s['annotation_leaks']}")
    print(f"Reports written to {Path(args.output_dir).resolve()}")


if __name__ == '__main__':
    configure_console()
    main()
