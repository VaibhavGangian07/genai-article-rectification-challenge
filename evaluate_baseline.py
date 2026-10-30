"""
BASELINE evaluation: annotation-stripped AI articles vs. human references.

This measures how far the AI text is from the human-rectified text when
nothing is corrected except removing the Error Annotations block. It is a
reference point for later phases, NOT a factual-accuracy score: word-level
similarity is high even when every factual error is still present.

Usage:
    python evaluate_baseline.py              # summary and changed text
    python evaluate_baseline.py --summary    # summary table only
"""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
from article_utils import strip_annotations, configure_console

ROOT = Path(__file__).resolve().parent
REFERENCE_DIR = ROOT / 'reference_rectified'
MANIFEST = REFERENCE_DIR / 'SHA256SUMS.txt'


def read_text(path: Path) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def verify_references():
    """Raise if any reference article differs from the recorded checksum."""
    for line in read_text(MANIFEST).splitlines():
        digest, name = line.split(maxsplit=1)
        data = (REFERENCE_DIR / name).read_bytes().replace(b'\r\n', b'\n')
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Reference article modified: {name}")


def compare(candidate: str, reference: str):
    """
    Word-level comparison of a candidate article against its reference.

    Returns:
        dict: similarity ratio and the list of changed spans
    """
    cand_words = candidate.split()
    ref_words = reference.split()
    matcher = difflib.SequenceMatcher(None, cand_words, ref_words, autojunk=False)
    changes = [
        {'op': op, 'candidate': ' '.join(cand_words[i1:i2]), 'reference': ' '.join(ref_words[j1:j2])}
        for op, i1, i2, j1, j2 in matcher.get_opcodes()
        if op != 'equal'
    ]
    return {
        'similarity': matcher.ratio(),
        'candidate_words': len(cand_words),
        'reference_words': len(ref_words),
        'changes': changes,
    }


def evaluate():
    """Compare the stripped AI article for every reference; returns per-article results."""
    verify_references()
    with open(ROOT / 'article_mapping.json', 'r') as f:
        mapping = {a['article_id']: a for a in json.load(f)}

    results = []
    for ref_path in sorted(REFERENCE_DIR.glob('article_*.txt')):
        article_id = ref_path.stem
        ai_text = read_text(ROOT / mapping[article_id]['ai_generated_file'])
        result = compare(strip_annotations(ai_text), read_text(ref_path))
        result['article_id'] = article_id
        results.append(result)
    return results


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison against human references (not a factual-accuracy score).")
    parser.add_argument('--summary', action='store_true', help='Print the summary table only')
    args = parser.parse_args()

    results = evaluate()

    print("=" * 70)
    print("BASELINE: annotation-stripped AI text vs. human references")
    print("Word-level similarity only. This is NOT a factual-accuracy score.")
    print("=" * 70)
    print(f"{'article':<14}{'similarity':>12}{'changed spans':>16}{'words (ai/ref)':>18}")
    for r in results:
        words = f"{r['candidate_words']}/{r['reference_words']}"
        print(f"{r['article_id']:<14}{r['similarity']:>12.4f}{len(r['changes']):>16}{words:>18}")
    mean = sum(r['similarity'] for r in results) / len(results)
    print(f"{'mean':<14}{mean:>12.4f}{sum(len(r['changes']) for r in results):>16}")

    if args.summary:
        return

    for r in results:
        print(f"\n--- {r['article_id']}: spans the human editor changed ---")
        for c in r['changes']:
            print(f"  [{c['op']}] AI:  {c['candidate']!r}")
            print(f"  {' ' * len(c['op'])}  REF: {c['reference']!r}")


if __name__ == "__main__":
    configure_console()
    main()
