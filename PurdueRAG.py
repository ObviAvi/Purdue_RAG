import json
from dotenv import load_dotenv
import os
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from semantic_text_splitter import TextSplitter
from rank_bm25 import BM25Okapi
import chromadb
from google import genai
from fastapi.middleware.cors import CORSMiddleware

# =========================
# CONFIG
# =========================
load_dotenv()

DATA_FILE = "reddit_posts_with_comments.json"
COLLECTION_NAME = "purdue_posts"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-2.5-flash"
API_KEY = os.getenv("GEMINI_API_KEY")

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

print("🔹 Splitting text into chunks...")
splitter = TextSplitter(capacity=512, overlap=64)
post_chunks = []
for index, post in enumerate(posts):
    text = post["title"] + " " + post.get("body", "")
    for chunk in splitter.chunks(text):
        post_chunks.append({"text": chunk, "metadata": {"index": index}})

print("🔹 Generating embeddings...")
model = SentenceTransformer(MODEL_NAME, device="cpu")
post_texts = [chunk["text"] for chunk in post_chunks]
post_embeddings = model.encode(post_texts, convert_to_numpy=True, device="cpu", normalize_embeddings=True)

print("🔹 Initializing ChromaDB...")
client = chromadb.Client()
if COLLECTION_NAME in [c.name for c in client.list_collections()]:
    collection = client.get_collection(COLLECTION_NAME)
else:
    collection = client.create_collection(COLLECTION_NAME)

# Add data to Chroma if empty
if len(collection.get()['ids']) == 0:
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

print("🔹 Setting up BM25 for hybrid retrieval...")
bm25 = BM25Okapi([text.split() for text in post_texts])

print("🔹 Connecting to Gemini...")
chat_client = genai.Client(api_key=API_KEY)

# =========================
# FUNCTIONS
# =========================
def get_context_from_posts(relevant_posts):
    context = ""
    for post in relevant_posts:
        context += f"Q: {post['title']}\n{post.get('body', '')}\n\n"
        if "comments" in post:
            for c in post["comments"]:
                context += f"score: {c.get('score', 0)}, answer: {c.get('body', '')}\n"
        context += "---\n"
    return context

'''
def hybrid_search(query, alpha=0.5, top_k=5):
    bm25_scores = np.array(bm25.get_scores(query.split()))
    query_emb = model.encode([query], normalize_embeddings=True)
    dense_scores = np.array(util.cos_sim(query_emb, post_embeddings)[0])

    # Normalize scores to [0, 1]
    bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-6)

    hybrid_scores = alpha * dense_scores + (1 - alpha) * bm25_scores
    top_idx = np.argsort(hybrid_scores)[::-1][:top_k]
    return [post_chunks[i] for i in top_idx]
'''

def retrieve_context(query):
    q_emb = model.encode([query])
    results = collection.query(query_embeddings=q_emb, n_results=5)
    idxs = [meta["index"] for meta in results["metadatas"][0]]
    relevant_posts = [posts[i] for i in idxs]

    for r in relevant_posts:
        print(r)

    return get_context_from_posts(relevant_posts)


def chat_with_rag(query):
    context = retrieve_context(query)
    prompt = f"""You are an experienced student at Purdue University. 
Use your experience and the following Reddit Q&A context to answer the question. 
Do not mention Reddit or the source explicitly.

Context:
{context}

Question: {query}
Answer:"""
    response = chat_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text

# =========================
# API
# =========================
app = FastAPI(title="Purdue Reddit RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8000"] for stricter control
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    answer = chat_with_rag(request.query)
    return {"query": request.query, "answer": answer}


@app.get("/")
def home():
    return {"message": "Purdue RAG API is running. Use POST /chat with {'query': '...'}."}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
