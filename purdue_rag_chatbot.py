import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
from starlette.staticfiles import StaticFiles
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
from google import genai

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

print("🔹 Connecting to persistent ChromaDB")
client = chromadb.PersistentClient(path="chroma_storage")
collection = client.get_collection(COLLECTION_NAME)

print("🔹 Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

print("🔹 Setting up BM25 for hybrid retrieval...")
post_texts = [doc for doc in collection.get()["documents"]]
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


def retrieve_context(query):
    q_emb = model.encode([query])
    results = collection.query(query_embeddings=q_emb, n_results=5)

    if not results["metadatas"] or not results["metadatas"][0]:
        print("⚠️ No results found in ChromaDB.")
        return "No relevant posts found."

    idxs = [meta["index"] for meta in results["metadatas"][0]]
    relevant_posts = [posts[i] for i in idxs]
    return get_context_from_posts(relevant_posts)


def chat_with_rag(query):
    print("🗣️ Query received:", query)
    context = retrieve_context(query)
    prompt = f"""Act as a student at Purdue University. 
                Use your experience and the following Reddit Q&A context to answer the question. 
                Do not mention Reddit or the source explicitly and do not mention that you are a student.
                Keep your answers detailed but not too long.

Context:
{context}

Question: {query}
Answer:"""
    response = chat_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


# =========================
# FASTAPI SETUP
# =========================
app = FastAPI(title="Purdue Reddit RAG API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
def chat_endpoint(request: QueryRequest):
    answer = chat_with_rag(request.query)
    return {"query": request.query, "answer": answer}

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=3000)
