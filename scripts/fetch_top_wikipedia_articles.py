#!/usr/bin/env python3
"""
Downloads the N most-viewed English Wikipedia articles as plain-text
files, one per article, for use as an offline text corpus.

Two Wikimedia APIs, both public and free, no API key needed:
  1. The Pageviews API to find which articles were most viewed on a
     given day (https://wikimedia.org/api/rest_v1/metrics/pageviews/...).
  2. The MediaWiki action API's "extracts" prop (from the TextExtracts
     extension) to fetch each article's plain-text extract - this is
     already fully rendered: templates ({{...}}), infoboxes, and other
     wikitext markup are expanded/stripped by MediaWiki's own parser
     before extraction, not left as raw wikitext for the caller to deal
     with. Output is readable prose, usable offline with no further
     processing.

Only Python's standard library is used - no pip install required.

Usage:
    python3 fetch_top_wikipedia_articles.py -n 50 -o ./corpus

Wikimedia's API etiquette requires a descriptive User-Agent with contact
info on every request, and asks that clients not hammer the API - both
are handled below (see USER_AGENT and REQUEST_DELAY_SECONDS).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Wikimedia asks every API client to identify itself - replace the contact
# email below with a real one before heavy/repeated use, per
# https://meta.wikimedia.org/wiki/User-Agent_policy
USER_AGENT = "llmdev-corpus-fetcher/1.0 (https://example.invalid/contact; contact@example.invalid)"

REQUEST_DELAY_SECONDS = 0.5  # Be polite - the API is shared infrastructure.

# Non-article pageview entries to skip (project/special pages, not content).
SKIP_TITLE_PREFIXES = ("Special:", "Main_Page", "Wikipedia:", "File:", "Portal:", "Help:", "Category:")


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def most_viewed_titles(language: str, date: datetime, count: int) -> list[str]:
    """Returns up to `count` article titles from the Pageviews API's top
    list for the given day, skipping non-article entries. Over-fetches
    (the API always returns ~1000 entries) so filtering doesn't leave the
    caller short."""
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/"
        f"{language}.wikipedia/all-access/{date:%Y/%m/%d}"
    )
    data = fetch_json(url)
    articles = data["items"][0]["articles"]

    titles: list[str] = []
    for article in articles:
        title = article["article"]
        if title.startswith(SKIP_TITLE_PREFIXES):
            continue
        titles.append(title)
        if len(titles) >= count:
            break
    return titles


def fetch_plain_text_extract(language: str, title: str) -> str | None:
    """Fetches the fully-rendered plain-text extract for one article -
    templates/infoboxes already expanded by MediaWiki's parser, not raw
    wikitext. Follows redirects. Returns None if the page doesn't exist
    or has no extractable text (e.g. a pure disambiguation/redirect
    stub)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = f"https://{language}.wikipedia.org/w/api.php?{urllib.parse.urlencode(params)}"
    data = fetch_json(url)

    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "missing" in page:
            return None
        extract = page.get("extract", "").strip()
        return extract or None
    return None


def safe_filename(title: str) -> str:
    """Wikipedia titles use underscores for spaces and can contain
    characters that aren't safe in filenames (/, :, etc.) - normalise to
    something every filesystem accepts."""
    name = title.replace("_", " ")
    name = re.sub(r'[\\/:*?"<>|]', "-", name)
    return name.strip()


_print_lock = threading.Lock()


def download_one(index: int, total: int, title: str, language: str, output_dir: Path) -> str:
    """Downloads (or skips) one article and returns which of "written",
    "already_present", or "skipped" it counts as. Print output is
    serialised via _print_lock so concurrent workers' lines don't
    interleave mid-line."""
    out_path = output_dir / f"{safe_filename(title)}.txt"
    if out_path.exists():
        with _print_lock:
            print(f"  [{index}/{total}] {title}... skipped (already downloaded)")
        return "already_present"

    try:
        text = fetch_plain_text_extract(language, title)
    except urllib.error.HTTPError as ex:
        with _print_lock:
            print(f"  [{index}/{total}] {title}... failed ({ex})")
        time.sleep(REQUEST_DELAY_SECONDS)
        return "skipped"

    if text is None:
        with _print_lock:
            print(f"  [{index}/{total}] {title}... skipped (no article text - redirect/disambiguation/missing)")
        time.sleep(REQUEST_DELAY_SECONDS)
        return "skipped"

    out_path.write_text(text, encoding="utf-8")
    with _print_lock:
        print(f"  [{index}/{total}] {title}... saved ({len(text):,} chars)")
    time.sleep(REQUEST_DELAY_SECONDS)
    return "written"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", "--count", type=int, default=50, help="Number of articles to fetch (default: 50).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("wikipedia_corpus"), help="Directory to write <Title>.txt files into.")
    parser.add_argument("-l", "--language", default="en", help="Wikipedia language edition (default: en).")
    parser.add_argument(
        "-w", "--workers", type=int, default=1,
        help="Number of articles to download concurrently (default: 1, sequential). Each worker still "
        "waits between its own requests, so total request rate scales with this - keep it modest "
        "(e.g. 4-8) to stay polite to the Wikimedia API.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Day to source the 'most viewed' list from, as YYYY-MM-DD (default: 2 days ago - "
        "the Pageviews API lags behind today's date by about a day).",
    )
    args = parser.parse_args()

    if args.workers < 1:
        print("--workers must be at least 1.", file=sys.stderr)
        return 1

    date = (
        datetime.strptime(args.date, "%Y-%m-%d")
        if args.date
        else datetime.now(timezone.utc) - timedelta(days=2)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Finding the {args.count} most-viewed {args.language}.wikipedia.org articles for {date:%Y-%m-%d}...")
    try:
        titles = most_viewed_titles(args.language, date, args.count)
    except urllib.error.HTTPError as ex:
        print(f"Failed to fetch the top-articles list: {ex}", file=sys.stderr)
        return 1

    if not titles:
        print("No article titles found for that day - try a different --date.", file=sys.stderr)
        return 1

    counts = {"written": 0, "already_present": 0, "skipped": 0}

    if args.workers == 1:
        for i, title in enumerate(titles, start=1):
            result = download_one(i, len(titles), title, args.language, args.output_dir)
            counts[result] += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(download_one, i, len(titles), title, args.language, args.output_dir): title
                for i, title in enumerate(titles, start=1)
            }
            for future in as_completed(futures):
                counts[future.result()] += 1

    written, already_present, skipped = counts["written"], counts["already_present"], counts["skipped"]
    print(
        f"\nDone: {written} article(s) written to {args.output_dir}/, "
        f"{already_present} already present (skipped), {skipped} skipped (no text/error)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
