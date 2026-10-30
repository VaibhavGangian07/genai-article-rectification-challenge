import sys
import json
import argparse
from pathlib import Path
from article_utils import strip_annotations, configure_console
from rectification_system import run

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

def save_rectified_article(article_id: str, rectified_content: str):
    mapping = get_article_mapping(article_id)
    fpath = mapping['rectified_file']
    
    # Ensure output directory exists
    output_path = Path(fpath)
    if output_path.resolve().is_relative_to(REFERENCE_DIR):
        raise ValueError(f"Refusing to overwrite reference article: {fpath}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(rectified_content)

def log_failure(article_id: str, message: str):
    print(f"[FAIL] {article_id}: {message}", file=sys.stderr)

def _rectify(article_id: str):
    """
    Rectify an article, falling back to the annotation-stripped AI text if
    the rectifier fails or returns nothing.
    
    Returns:
        tuple: (rectified content, True if the fallback was used)
    """
    ai_generated_content = get_ai_generated_article(article_id)
    
    try:
        # PLUG YOUR CUSTOM RECTIFIER HERE
        rectified_content = run(ai_generated_content)
        ###################################
        if not isinstance(rectified_content, str) or not rectified_content.strip():
            raise ValueError("rectifier returned empty output")
        used_fallback = False
    except Exception as e:
        log_failure(article_id, f"{type(e).__name__}: {e} -- writing annotation-stripped fallback")
        rectified_content = strip_annotations(ai_generated_content)
        used_fallback = True
    
    save_rectified_article(article_id, rectified_content)
    
    print(f"[{'FALLBACK' if used_fallback else 'OK'}] Rectified {article_id}")
    return rectified_content, used_fallback

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
        dict: article IDs grouped by outcome ('ok', 'fallback', 'failed')
    """
    outcomes = {'ok': [], 'fallback': [], 'failed': []}
    total = len(articles)
    
    for i, article in enumerate(articles):
        article_id = article['article_id']
        
        print(f"\nProcessing {article_id} ({i+1}/{total})...")
        
        try:
            _, used_fallback = _rectify(article_id)
            outcomes['fallback' if used_fallback else 'ok'].append(article_id)
        except Exception as e:
            log_failure(article_id, f"{type(e).__name__}: {e} -- no output written")
            outcomes['failed'].append(article_id)
    
    print(f"\n{'='*50}")
    print(f"Completed! Processed {total} articles.")
    print(f"  Rectified: {len(outcomes['ok'])}")
    print(f"  Fallback:  {len(outcomes['fallback'])} {' '.join(outcomes['fallback'])}")
    print(f"  Failed:    {len(outcomes['failed'])} {' '.join(outcomes['failed'])}")
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


if __name__ == "__main__":
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
    
    args = parser.parse_args()
    
    if args.command == 'test':
        print(f"Testing rectification system on first {args.count} articles...")
        test_rectifier(count=args.count)
    elif args.command == 'rectify-all':
        print("Processing all 104 articles...")
        rectify_all()