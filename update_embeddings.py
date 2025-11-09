import json
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from semantic_text_splitter import TextSplitter
import chromadb

# =========================
# CONFIG
# =========================
load_dotenv()

DATA_FILE = "reddit_posts_with_comments.json"
COLLECTION_NAME = "purdue_posts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# =========================
# INITIALIZATION
# =========================
print("🔹 Loading posts...")
with open(DATA_FILE, "r") as f:
    posts = json.load(f)

# Sort comments by score for each post
for post in posts:
    if "comments" in post:
        post["comments"].sort(key=lambda x: x.get("score", 0), reverse=True)

print("🔹 Splitting posts into chunks...")
splitter = TextSplitter(capacity=512, overlap=64)
post_chunks = []
for index, post in enumerate(posts):
    text = post["title"] + " " + post.get("body", "")
    for chunk in splitter.chunks(text):
        post_chunks.append({"text": chunk, "metadata": {"index": index}})

print(f"Total chunks: {len(post_chunks)}")

print("🔹 Generating embeddings...")
model = SentenceTransformer(MODEL_NAME, device="cpu")
post_texts = [chunk["text"] for chunk in post_chunks]
post_embeddings = model.encode(post_texts, convert_to_numpy=True, device="cpu", normalize_embeddings=True)

print("🔹 Initializing ChromaDB...")
client = chromadb.PersistentClient(path="chroma_storage")

# Create or overwrite the collection
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    print("⚠️ Deleting existing collection to rebuild embeddings.")
    client.delete_collection(COLLECTION_NAME)

collection = client.create_collection(COLLECTION_NAME)

print("🔹 Populating ChromaDB...")
batch_size = 5000
for i in range(0, len(post_chunks), batch_size):
    batch_emb = post_embeddings[i:i+batch_size]
    batch_docs = [c["text"] for c in post_chunks[i:i+batch_size]]
    batch_meta = [c["metadata"] for c in post_chunks[i:i+batch_size]]
    batch_ids = [str(i+j) for j in range(len(batch_docs))]
    collection.add(
        embeddings=batch_emb.tolist(),
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids,
    )

print(f"Embeddings updated and stored in persistent ChromaDB ({len(collection.get()['ids'])} entries).")
