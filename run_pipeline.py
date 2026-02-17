#!/usr/bin/env python3
"""
DSCI-560 Lab 5: Encompassing script for periodic scrape → preprocess → storage,
and for keyword/message search with closest-cluster display and graphical representation.

Usage:
  python run_pipeline.py <interval_minutes>   -- run update every N minutes; between runs, prompt for search
  python run_pipeline.py                     -- search-only mode (prompt for keywords/message)

Example:
  python run_pipeline.py 5     → fetch, process, and update storage every 5 minutes
  python run_pipeline.py       → enter keywords to find closest cluster and see messages + graph
"""
import sys
import time
import argparse
import os
import csv


def log(msg):
    print(msg, flush=True)


def _load_existing_csv(csv_path: str):
    """Load existing CSV rows as post-like dicts (id, title, ...) for merging. Returns [] if file missing."""
    if not os.path.isfile(csv_path):
        return []
    out = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("post_id", "").strip()
            if not pid:
                continue
            try:
                created_utc = int(float(row.get("created_utc") or 0))
            except (TypeError, ValueError):
                created_utc = row.get("created_utc", "")
            image_urls = (row.get("image_urls") or "").strip()
            perm = (row.get("permalink") or "").replace("https://old.reddit.com", "").replace("http://old.reddit.com", "")
            out.append({
                "id": pid,
                "title": row.get("title", ""),
                "author": row.get("author", ""),
                "subreddit": row.get("subreddit", ""),
                "score": int(row.get("score") or 0),
                "num_comments": int(row.get("num_comments") or 0),
                "created_utc": created_utc,
                "created_iso": row.get("created_iso", ""),
                "scraped_datetime": row.get("scraped_datetime", ""),
                "url": row.get("url", ""),
                "permalink": perm,
                "json_url": row.get("json_url", ""),
                "selftext": row.get("selftext", ""),
                "_image_urls": [u.strip() for u in image_urls.split("|") if u.strip()] if image_urls else [],
            })
    return out


def run_one_update(subreddit: str, num_posts: int, csv_path: str) -> bool:
    """
    Run one full cycle: fetch data → merge with existing CSV (all available data) → process data → update storage.
    data_processing_and_analysis runs on the full merged CSV so clustering uses all documents, not just the latest fetch.
    Returns True on success, False on failure. Prints clear status and error messages.
    """
    log("\n" + "=" * 60)
    log("Fetching data...")
    try:
        from reddit_scraper import fetch_posts, save_to_csv
        existing = _load_existing_csv(csv_path)
        existing_ids = {p["id"] for p in existing}
        log(f"Existing data: {len(existing)} posts in CSV.")
        posts = fetch_posts(subreddit, num_posts)
        if not posts:
            log("ERROR: No posts fetched. Check subreddit name and network.")
            return False
        new_posts = [p for p in posts if p.get("id") not in existing_ids]
        combined = existing + new_posts
        save_to_csv(combined, csv_path)
        log(f"Fetching data: retrieved {len(posts)} new posts; {len(new_posts)} added. Total in CSV: {len(combined)}.")
    except FileNotFoundError as e:
        log(f"ERROR: Could not import reddit_scraper. {e}")
        return False
    except Exception as e:
        log(f"ERROR (fetching data): {e}")
        return False

    log("\nProcessing data...")
    try:
        import data_processing_and_analysis
        data_processing_and_analysis.run_ocr(csv_path=csv_path)
        data_processing_and_analysis.run_embedding(csv_path=csv_path)
        data_processing_and_analysis.run_cluster_tree(csv_path=csv_path)
        log("Processing data: completed (OCR, embedding, clustering).")
    except Exception as e:
        log(f"ERROR (processing data): {e}")
        return False

    log("\nDatabase / storage updated.")
    log("=" * 60 + "\n")
    return True


def run_search_mode():
    """Prompt for keywords or message, find closest cluster, display messages and graph."""
    log("Search mode: enter keywords or a message to find the closest cluster.")
    log("(Run with an interval first to populate data, e.g. python run_pipeline.py 5)")
    log("Enter 'quit' or 'exit' to stop.\n")

    try:
        import data_processing_and_analysis
        import os
        vectors_path = os.path.join("data", "processed_data", "doc_vectors.npy")
        labels_path = os.path.join("data", "processed_data", "cluster_labels.npy")
        if not os.path.isfile(vectors_path) or not os.path.isfile(labels_path):
            log("ERROR: No cluster data found. Run with an interval first: python run_pipeline.py 5")
            return
    except Exception as e:
        log(f"ERROR: {e}")
        return

    while True:
        try:
            query = input("Keywords or message: ").strip()
        except (EOFError, KeyboardInterrupt):
            log("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            log("Exiting.")
            break
        try:
            data_processing_and_analysis.find_closest_cluster_and_display(query)
        except Exception as e:
            log(f"ERROR (search): {e}")


def run_daemon(interval_min: int, subreddit: str, num_posts: int, csv_path: str):
    """Run update every interval_min minutes; between runs, prompt for search."""
    log(f"Running pipeline every {interval_min} minute(s). Subreddit: r/{subreddit}, posts per run: {num_posts}")
    log("Press Ctrl+C to stop. Between updates you can enter keywords to search.\n")

    run_count = 0
    while True:
        run_count += 1
        log(f"--- Update run #{run_count} ---")
        ok = run_one_update(subreddit, num_posts, csv_path)
        if not ok:
            log("Update failed. Will retry at next interval.")

        wait_sec = interval_min * 60
        log(f"Next update in {interval_min} minute(s). (Run 'python run_pipeline.py' in another terminal to search by keywords.)")
        try:
            time.sleep(wait_sec)
        except KeyboardInterrupt:
            log("\nStopped by user.")
            break


def main():
    parser = argparse.ArgumentParser(
        description="DSCI-560: Periodic scrape → process → storage; or search by keywords for closest cluster."
    )
    parser.add_argument(
        "interval_minutes",
        nargs="?",
        type=int,
        default=None,
        help="Run update every N minutes (omit for search-only mode)",
    )
    parser.add_argument(
        "--subreddit",
        default="tech",
        help="Subreddit to scrape (default: tech)",
    )
    parser.add_argument(
        "--posts",
        type=int,
        default=200,
        help="Number of posts to fetch per update (default: 200)",
    )
    parser.add_argument(
        "--csv",
        default="reddit_posts.csv",
        help="Output CSV path (default: reddit_posts.csv)",
    )
    args = parser.parse_args()

    if args.interval_minutes is not None:
        if args.interval_minutes <= 0:
            log("Interval must be positive. Use search-only mode by not passing an interval.")
            run_search_mode()
        else:
            run_daemon(args.interval_minutes, args.subreddit, args.posts, args.csv)
    else:
        run_search_mode()


if __name__ == "__main__":
    main()
