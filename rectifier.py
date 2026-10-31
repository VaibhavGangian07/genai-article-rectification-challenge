import sys
import json
import argparse
from pathlib import Path
from article_utils import strip_annotations, find_annotation_leak, configure_console
from rectification_system import run

# Article outcomes; only 'rectified' counts as a successful correction
STATUSES = ('rectified', 'no_change', 'no_valid_edits', 'fallback', 'failed')

# Human-rectified reference articles; never written to by the pipeline
REFERENCE_DIR = Path(__file__).resolve().parent / 'reference_rectified'

def get_article_mapping(article_id: str):
    # Load article mapping to get file paths
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    # Find the article by ID
    article_data = next((a for a in articles if a['article_id'] == article_id), None)
    if not article_data:
        raise ValueError(f"Article {article_id} not found in mapping")
    
    return article_data

def get_ai_generated_article(article_id: str):
    # Read the AI-generated article
    _mapping = get_article_mapping(article_id)
    fpath = _mapping['ai_generated_file']
    with open(fpath, 'r', encoding='utf-8') as f:
        article = f.read()
    return article

def get_source_article(article_id: str):
    # Read the source article the AI article was generated from
    fpath = get_article_mapping(article_id)['source_file']
    with open(fpath, 'r', encoding='utf-8') as f:
        return f.read()

def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    fpath = mapping['rectified_file']

    # Ensure output directory exists
    output_path = Path(fpath)
    if output_path.resolve().is_relative_to(REFERENCE_DIR):
        raise ValueError(f"Refusing to overwrite reference article: {fpath}")
    leak = find_annotation_leak(rectified_content)
    if leak:
        raise ValueError(f"Refusing to save annotation text ({leak!r}) in {fpath}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(rectified_content)

def log_failure(article_id: str, message: str):
    print(f"[FAIL] {article_id}: {message}", file=sys.stderr)

def _rectify(article_id: str):
    """
    Rectify an article, falling back to the annotation-stripped AI text if
    the rectifier fails, returns nothing, or returns annotation text.

    Returns:
        tuple: (rectified content, status, reason, usage)
    """
    ai_generated_content = get_ai_generated_article(article_id)
    usage = {}

    try:
        source_content = get_source_article(article_id)
        # PLUG YOUR CUSTOM RECTIFIER HERE
        result = run(ai_generated_content, source_content, article_id)
        ###################################
        usage = getattr(result, 'usage', None) or {}
        rectified_content, status, reason = result.text, result.status, result.reason
        if status not in STATUSES or status == 'failed':
            raise ValueError(f"rectifier returned unknown status {status!r}")
        if not isinstance(rectified_content, str) or not rectified_content.strip():
            raise ValueError("rectifier returned empty output")
        leak = find_annotation_leak(rectified_content)
        if leak:
            raise ValueError(f"rectifier output contains annotation text ({leak!r})")
    except Exception as e:
        rectified_content, status = strip_annotations(ai_generated_content), 'fallback'
        reason = f"{type(e).__name__}: {e}"

    if status == 'fallback':
        log_failure(article_id, f"{reason} -- writing annotation-stripped fallback")

    save_rectified_article(article_id, rectified_content)

    print(f"[{status.upper()}] {article_id}" + (f" ({reason})" if reason and status != 'fallback' else ""))
    return rectified_content, status, reason, usage

def rectify_article(article_id: str):
    """
    Rectify an AI-generated article.
    
    Args:
        article_id: ID of the article (e.g., 'article_001')
    
    Returns:
        str: The rectified article content
    """
    return _rectify(article_id)[0]


def process_articles(articles: list):
    """
    Rectify each article in turn, continuing past failures.
    
    Returns:
        dict: article IDs grouped by status (see STATUSES); each article appears exactly once
    """
    outcomes = {status: [] for status in STATUSES}
    reasons = {}
    totals = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'cost_usd': 0.0}
    article_ids = list(dict.fromkeys(article['article_id'] for article in articles))
    if len(article_ids) != len(articles):
        print(f"Skipping {len(articles) - len(article_ids)} duplicate article entries")
    total = len(article_ids)

    for i, article_id in enumerate(article_ids):
        print(f"\nProcessing {article_id} ({i+1}/{total})...")

        try:
            _, status, reason, usage = _rectify(article_id)
        except Exception as e:
            log_failure(article_id, f"{type(e).__name__}: {e} -- no output written")
            status, reason, usage = 'failed', f"{type(e).__name__}: {e}", {}
        outcomes[status].append(article_id)
        if status in ('fallback', 'failed'):
            reasons[article_id] = reason
        for key in totals:
            totals[key] += usage.get(key, 0) or 0

    assert sum(len(ids) for ids in outcomes.values()) == total, "every article must have exactly one status"

    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles.")
    print(f"  Rectified:      {len(outcomes['rectified'])}")
    print(f"  No change:      {len(outcomes['no_change'])} (no errors proposed)")
    print(f"  No valid edits: {len(outcomes['no_valid_edits'])} (all proposed edits rejected)")
    print(f"  Fallback:       {len(outcomes['fallback'])} (annotation-stripped text, not corrected)")
    print(f"  Failed:         {len(outcomes['failed'])} (no output written)")
    for article_id, reason in reasons.items():
        print(f"    {article_id}: {reason}")
    print(f"  LLM usage:      {totals['calls']} requests, {totals['prompt_tokens']} prompt + "
          f"{totals['completion_tokens']} completion tokens, ${totals['cost_usd']:.4f} recorded")
    print(f"{'='*50}")
    return outcomes


def test_rectifier(count: int):
    """
    Test the rectification system on a subset of articles.
    
    Args:
        count: Number of articles to test (default: 16)
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    # Test on first 'count' articles
    return process_articles(articles[:count])


def rectify_all():
    """
    Generate rectified articles for all articles in the mapping.
    """
    # Load article mapping
    with open('article_mapping.json', 'r') as f:
        articles = json.load(f)
    
    return process_articles(articles)


def main(argv=None) -> int:
    """Run the CLI; the exit code is 1 if any article fell back or failed, else 0."""
    configure_console()
    
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles by fixing errors and inaccuracies."
    )
    parser.add_argument(
        'command',
        choices=['test', 'rectify-all'],
        help='Command to execute: "test" to process first 16 articles, "rectify-all" to process all 104 articles'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=16,
        help='Number of articles to test (only applicable for "test" command, default: 16)'
    )
    
    args = parser.parse_args(argv)
    
    if args.command == 'test':
        print(f"Testing rectification system on first {args.count} articles...")
        outcomes = test_rectifier(count=args.count)
    elif args.command == 'rectify-all':
        print("Processing all 104 articles...")
        outcomes = rectify_all()
    return 1 if outcomes['fallback'] or outcomes['failed'] else 0


if __name__ == "__main__":
    sys.exit(main())
