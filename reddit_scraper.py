import requests
import time
import csv
from datetime import datetime, timezone
from bs4 import BeautifulSoup

OLD_REDDIT_BASE = "https://old.reddit.com"
# Using old reddit since it is much more leniet than new reddit

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def fetch_posts(subreddit, count=100):
    # Getting posts from a subreddit using old Reddit's json API
    # By default it gets 100 posts from that subreddit
    posts = []
    after = None

    while len(posts) < count:
        # we stop when we reach the count of posts we wnana scrape
        batch_size = min(100, count - len(posts))
        url = f"{OLD_REDDIT_BASE}/r/{subreddit}/new.json"
        params = {"limit": batch_size}
        if after:
            params["after"] = after

        for attempt in range(5):
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            # if error, wait for 3 minutes
            if resp.status_code == 429:
                print(f" The Rate is limited! Waiting 3 minutes (attempt {attempt+1}/5)")
                time.sleep(180)
                continue
            resp.raise_for_status()
            break
        else:
            print(" Failed after 5 retries. Returning what we have :(")
            break
        data = resp.json().get("data", {})
        children = data.get("children", [])

        if not children:
            print(f"No more posts are available. Stopped at {len(posts)} posts.")
            break

        for child in children:
            posts.append(child["data"])

        after = data.get("after")
        if not after:
            print(f"Reached end of subreddit. Got {len(posts)} posts.")
            break

        print(f"Fetched {len(posts)}/{count} posts...")

        # Pause for 1 minute every 1000 posts to avoid Reddit's rate limit
        if len(posts) % 1000 == 0:
            print(f"\n--- Reached {len(posts)} posts. Waiting 60 seconds to avoid rate limit ---\n")
            time.sleep(60)
        else:
            time.sleep(1)
    return posts



def scrape_images_from_old_reddit(permalink):
    # Scrape actual working image URLs from an old Reddit post page
    images = []
    
    # using the same 5 trial and wait when error method        
    for attempt in range(5):
    
        try:
            resp = requests.get(permalink, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                print(f" The Rate is limited! Waiting 3 minutes (attempt {attempt+1}/5)")
                time.sleep(180)
                continue
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            for a in soup.select(".expando a[href]"):
                href = a["href"]
                if "preview.redd.it" in href or "i.redd.it" in href:
                    images.append(href)

            if not images:
                for img in soup.select(".expando img[src]"):
                    src = img["src"]
                    if "preview.redd.it" in src or "i.redd.it" in src:
                        images.append(src)

            return images

        except requests.exceptions.HTTPError:
            print(f"    HTTP error! Waiting 3 minutes (attempt {attempt+1}/5)")
            time.sleep(180)
        except Exception as e:
            print(f"    Error scraping images: {e}")
            return images

    return images


def has_images(post):
    # just checking if a post has a image or not, no scarpgin
    if post.get("is_gallery"):
        return True
    post_url = post.get("url", "")
    if post_url.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
        return True
    if "i.redd.it" in post_url or "i.imgur.com" in post_url:
        return True
    if post.get("post_hint") == "image":
        return True
    return False


def utc_to_iso(utc_timestamp):
    # conert the utc timestampt to iso
    return datetime.fromtimestamp(utc_timestamp, tz=timezone.utc).isoformat()

def save_to_csv(posts, filename="reddit_posts.csv"):
    fields = [
        "post_id", "title", "author", "subreddit", "score", "num_comments",
        "created_utc", "created_iso", "scraped_datetime", "url", "permalink", "json_url",
        "selftext", "image_urls",
    ]
    scraped_now = datetime.now(timezone.utc).isoformat()

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for post in posts:
            # Preserve scraped_datetime for rows loaded from CSV; set to now for newly scraped
            scraped_dt = post.get("scraped_datetime") or scraped_now
            permalink = post.get("permalink", "")
            if permalink and not permalink.startswith("http"):
                permalink = f"{OLD_REDDIT_BASE}{permalink}"
            row = {
                "post_id": post.get("id", ""),
                "title": post.get("title", ""),
                "author": post.get("author", ""),
                "subreddit": post.get("subreddit", ""),
                "score": post.get("score", 0),
                "num_comments": post.get("num_comments", 0),
                "created_utc": post.get("created_utc", ""),
                "created_iso": post.get("created_iso") or utc_to_iso(post.get("created_utc", 0)),
                "scraped_datetime": scraped_dt,
                "url": post.get("url", ""),
                "permalink": permalink or f"{OLD_REDDIT_BASE}{post.get('permalink', '')}",
                "json_url": post.get("json_url") or f"{OLD_REDDIT_BASE}{post.get('permalink', '')}.json",
                "selftext": post.get("selftext", ""),
                "image_urls": " | ".join(post.get("_image_urls", [])),
            }
            writer.writerow(row)

    print(f"Saved {len(posts)} posts to {filename}")

######################################################################
###### Main function:
######################################################################
if __name__ == "__main__":
    # List of subreddits to scrape (moves to next when current one runs out of posts or goes until we hit 5k):
    subreddits = ["badroommates", "roommateproblems", "roommatesfromhell", "RoommateStories", "roommates", "movingout", "Tenant"]

    # Total number of posts we want to scrape across all subreddits:
    count = 5000

    all_posts = []
    remaining = count

    for sub in subreddits:
        if remaining <= 0:
            break

        print(f"\n{'='*70}")
        print(f"Scraping r/{sub} (need {remaining} more posts)...\n")
        posts = fetch_posts(sub, remaining)
        print(f"Got {len(posts)} posts from r/{sub}")

        # Scrape images for posts that have them
        image_posts = [p for p in posts if has_images(p)]
        if image_posts:
            print(f"Found {len(image_posts)} posts with images. Scraping image URLs\n")
            for i, post in enumerate(image_posts, 1):
                permalink = f"{OLD_REDDIT_BASE}{post['permalink']}"
                print(f"  [{i}/{len(image_posts)}] Scraping: {post['title'][:50]}")
                post["_image_urls"] = scrape_images_from_old_reddit(permalink)
                time.sleep(2)

        all_posts.extend(posts)
        remaining -= len(posts)

    print(f"\n{'='*80}")
    print(f"Total posts collected: {len(all_posts)}")
    # save as csv
    save_to_csv(all_posts)
