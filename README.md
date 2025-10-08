# RAG-Powered Purdue-Specific Data Search

This project implements an end-to-end Retrieval-Augmented Generation (RAG) system for answering Purdue-specific questions. It includes:

* A **web scraper** to collect data from the Purdue subreddit
* An **embedding generator** to create semantic representations for each scraped post and comments
* A **frontend UI** to query and display relevant results

---

## Project Overview

**Architecture Flow**

1. **Scraper (`scraper.py`)** – Extracts text content, headings, and metadata from company websites.
2. **Embedding Generator (`generate_embeddings.py`)** – Converts the scraped data into vector embeddings using SentenceTransformers. The embeddings are stored in a local file or vector database.
3. **Frontend (`index.html`)** – A simple HTML/JS interface that sends a query to the backend and displays relevant results.
4. **Backend (`app.py`)** – A lightweight FastAPI or Flask server that receives frontend queries, retrieves semantically similar chunks using embeddings, and returns the results.

---

## Setup Instructions

### 1. Clone and Install Dependencies

```bash
git clone <your_repo_url>
cd project
pip install -r requirements.txt
```

### 2. Scrape Manufacturing Websites

Edit `scraper.py` to include the URLs you want to crawl, then run:

```bash
python scraper.py
```

Output json files will appear in root folder.

### 3. Generate Embeddings

Once text data is ready, generate embeddings with:

```bash
python generate_embeddings.py
```

This creates a `.pkl` or `.json` file in `data/embeddings/` containing the text chunks and their vector representations.

### 4. Start the Backend

Run the RAG server (Flask or FastAPI):

```bash
python app.py
```

By default, it runs on `http://127.0.0.1:8000`.

### 5. Open the Frontend

Simply open `index.html` in your browser.
If you’re running a local server, make sure it points to the correct backend endpoint, e.g.:

```javascript
fetch('http://127.0.0.1:8000/query', {...})
```

---

## 🔍 Example Query Flow

1. User types: *"What is the best dining court at Purdue"*
2. Frontend sends the query to `/query` endpoint.
3. Backend embeds the query → retrieves top-k similar chunks and passes the retrived Purdue subreddit context to an LLM.
4. Results (and optionally a generated summary) are displayed on the frontend.

---

## ⚙️ Configuration

You can customize parameters in `generate_embeddings.py`:

* `embedding_model`: change between OpenAI, MiniLM, or Instructor models
* `chunk_size`: control how large text fragments are before embedding
* `vector_db`: choose between FAISS, Chroma, or in-memory search

---

## 🧰 Example Requirements

Example `requirements.txt`:

```
requests
beautifulsoup4
tqdm
openai
sentence-transformers
faiss-cpu
flask
```

---

## Future Improvements

* Add authentication and upload interface for new documents
* Integrate LangChain for more advanced RAG pipelines
* Deploy on Streamlit or Next.js frontend
* Add caching for faster repeated queries
