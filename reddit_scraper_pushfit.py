import requests
import time
from tqdm import tqdm
import json

BASE_URL = "https://api.pushshift.io/reddit"

def fetch_pushshift(endpoint, params):
    url = f"{BASE_URL}/{endpoint}/"
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        print("Error", r.status_code)
        return []
    data = r.json().get("data")
    if not data:
        return []
    return data

def get_submissions(subreddit, limit=1000):
    all_posts = []
    before = int(time.time())
    print(f"Fetching submissions from r/{subreddit} ...")
    while len(all_posts) < limit:
        params = {
            "subreddit": subreddit,
            "size": 100,
            "before": before,
            "sort": "desc",
            "sort_type": "created_utc",
        }
        data = fetch_pushshift("submission/search", params)
        if not data:
            break
        all_posts.extend(data)
        before = data[-1]["created_utc"]
        time.sleep(1)
    return all_posts[:limit]

def get_comments_for_post(post_id):
    params = {"link_id": post_id, "size": 500}
    comments = fetch_pushshift("comment/search", params)
    return [
        {
            "id": c["id"],
            "body": c.get("body", ""),
            "score": c.get("score", 0),
            "created": c.get("created_utc", 0.0),
        }
        for c in comments
    ]

def build_dataset(subreddit, post_limit=100):
    submissions = get_submissions(subreddit, post_limit)
    dataset = []
    for sub in tqdm(submissions):
        post_id = sub["id"]
        comments = get_comments_for_post(post_id)
        dataset.append({
            "id": post_id,
            "title": sub.get("title", ""),
            "body": sub.get("selftext", ""),
            "url": sub.get("full_link", ""),
            "score": sub.get("score", 0),
            "created": sub.get("created_utc", 0.0),
            "comments": comments
        })
        time.sleep(0.5)
    return dataset

if __name__ == "__main__":
    data = build_dataset("purdue", post_limit=50)
    with open("purdue_subreddit.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} posts to purdue_subreddit.json")
