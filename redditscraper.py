# reddit_scraper.py
from praw import Reddit
import json
import time

# Replace these with your Reddit API credentials
# (You can get them from https://www.reddit.com/prefs/apps)
reddit = Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="purdue_scraper by u/YOUR_USERNAME"
)

data = []

# Use "hot", "new", or "top"
subreddit = reddit.subreddit("purdue")

print("🔹 Scraping posts from r/purdue...")
for post in subreddit.hot(limit=1000):
    post.comments.replace_more(limit=0)  # fetch all comments
    comments = []
    for comment in post.comments.list():
        comments.append({
            "body": comment.body,
            "score": comment.score
        })
    data.append({
        "title": post.title,
        "body": post.selftext,
        "url": post.url,
        "score": post.score,
        "created": post.created_utc,
        "comments": comments
    })
    time.sleep(0.5)  # respect Reddit’s rate limits

with open("reddit_posts.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"✅ Scraped {len(data)} posts and saved to reddit_posts.json")
