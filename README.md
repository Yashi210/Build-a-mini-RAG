# 🏗️ Construction AI Assistant — Mini RAG

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for a construction marketplace, built with Python, LangChain, FAISS, Sentence Transformers, and Streamlit.

---

## Architecture

```
User Query
    │
    ▼
[Streamlit UI]  ──► uploads PDFs/TXTs
    │
    ▼
[Document Parser]  pdfplumber / utf-8
    │
    ▼
[LangChain Splitter]  RecursiveCharacterTextSplitter
    │  chunk_size=400, chunk_overlap=50
    ▼
[Sentence Transformers]  all-MiniLM-L6-v2 (local, free)
    │  384-dim normalized embeddings
    ▼
[FAISS IndexFlatIP]  cosine similarity search
    │
    ▼  top-k chunks
[OpenRouter LLM]  mistral-7b / llama-3 / gemma-3
    │  strict grounding prompt
    ▼
[Streamlit Chat UI]  answer + retrieved context + metrics
```

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| **UI** | Streamlit | Fast, Python-native, easy to deploy |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Respects sentence/paragraph boundaries |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Free, fast, 384-dim, runs locally |
| **Vector store** | FAISS `IndexFlatIP` | No external service, exact cosine search |
| **LLM** | OpenRouter (Mistral-7B free tier) | Free, powerful, grounded generation |
| **PDF parsing** | `pdfplumber` | Accurate text extraction from PDFs |

---

## Quickstart (local)

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/construction-rag.git
cd construction-rag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Get a free OpenRouter API key

1. Go to [openrouter.ai](https://openrouter.ai) and sign up (free)
2. Create an API key
3. Either paste it in the sidebar when the app runs, **or** set it as an environment variable:

```bash
export OPENROUTER_API_KEY=sk-or-YOUR_KEY_HERE
```

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Use the app

1. Upload one or more PDF or TXT documents in the sidebar
2. Adjust chunk size, overlap, and top-k if needed
3. Click **Build / Rebuild Index**
4. Ask questions in the chat — see retrieved context + generated answer

---

## Project Structure

```
construction-rag/
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # Core RAG logic (chunking, FAISS, LLM)
├── requirements.txt
├── render.yaml             # Render deployment config
├── .streamlit/
│   └── config.toml         # Theme + server settings
└── sample_docs/
    └── construction_faq.txt  # Demo document to test with
```

---

## Deployment

### Option A — Render (recommended, free tier available)

1. Push your repo to GitHub
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render auto-detects `render.yaml` — click **Deploy**
5. In the Render dashboard → **Environment** → add:
   ```
   OPENROUTER_API_KEY = sk-or-YOUR_KEY
   ```
6. Your app will be live at `https://your-app-name.onrender.com`

> **Note:** Render's free tier spins down after 15 min of inactivity. First request after sleep takes ~30s.

### Option B — Vercel (via `vercel-python` or Docker)

Vercel is primarily for serverless frontend apps. Streamlit requires a persistent server, so use this approach:

1. Add a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. Deploy to Vercel using their Docker/container support, or use **Railway** (simpler for containers):

```bash
npm install -g railway
railway login
railway init
railway up
```

### Option C — Streamlit Community Cloud (easiest, free)

1. Push to a **public** GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo and `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENROUTER_API_KEY = "sk-or-YOUR_KEY"
   ```
5. Click **Deploy** — live in ~2 minutes!

---

## How grounding is enforced

The LLM receives this system prompt:

```
You are a helpful AI assistant for a construction marketplace.
Answer the user's question ONLY using the provided context below.
If the answer cannot be found in the context, say:
"I couldn't find information about this in the provided documents."
Do NOT use any external knowledge. Be concise and factual.
```

Retrieved chunks are injected directly into the user prompt as `[Source: filename]` blocks. The model is instructed to refuse answering anything not present in those blocks.

---

## Quality Observations (sample evaluation)

| # | Question | Chunk relevance | Hallucination | Completeness |
|---|---|---|---|---|
| 1 | What causes project delays? | ✅ High | ❌ None | ✅ Complete |
| 2 | What PPE is required? | ✅ High | ❌ None | ✅ Complete |
| 3 | How are payments processed? | ✅ High | ❌ None | ✅ Complete |
| 4 | What is the concrete strength standard? | ✅ High | ❌ None | ✅ Complete |
| 5 | What is the bid bond requirement? | ✅ High | ❌ None | ✅ Complete |
| 6 | How long is the defects liability period? | ✅ High | ❌ None | ✅ Complete |
| 7 | What is the capital of France? | ⬜ None | ❌ Refuses | ✅ Correct refusal |

**Finding:** Grounding is reliable when questions match document content. The model correctly refuses out-of-scope questions. Chunk size of 400 tokens balances context richness vs. noise.

---

## License

MIT
