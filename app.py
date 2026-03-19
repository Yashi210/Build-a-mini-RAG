import streamlit as st
import os
import time
from rag_pipeline import RAGPipeline

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction AI Assistant",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #c8dff5; margin: 0.3rem 0 0; font-size: 1rem; }

    .context-card {
        background: #f0f7ff;
        border-left: 4px solid #2d6a9f;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #1a1a2e;
    }
    .context-card .chunk-meta {
        font-weight: 600;
        color: #2d6a9f;
        margin-bottom: 0.3rem;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .answer-card {
        background: #f6fff8;
        border-left: 4px solid #27ae60;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        margin-top: 1rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 0.8rem 0;
    }
    .metric-pill {
        background: #e8f4fd;
        border: 1px solid #b3d7f5;
        border-radius: 20px;
        padding: 0.2rem 0.7rem;
        font-size: 0.78rem;
        color: #1a5276;
        font-weight: 500;
    }
    .stChatMessage { border-radius: 10px; }
    div[data-testid="stExpander"] { border: 1px solid #dce8f5; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏗️ Construction AI Assistant</h1>
    <p>RAG-powered Q&A grounded in your internal construction documents</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        help="Get a free key at openrouter.ai",
    )

    st.divider()
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF / TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload construction policies, FAQs, or specs",
    )

    st.divider()
    st.subheader("🔧 Retrieval Settings")
    top_k = st.slider("Top-K chunks", min_value=1, max_value=8, value=3,
                      help="Number of document chunks to retrieve per query")
    chunk_size   = st.slider("Chunk size (tokens)", 200, 800, 400, step=50)
    chunk_overlap = st.slider("Chunk overlap (tokens)", 0, 200, 50, step=25)

    st.divider()
    st.subheader("🤖 Model")
    model_choice = st.selectbox(
        "LLM",
        [
            "mistralai/mistral-7b-instruct:free",
            "meta-llama/llama-3-8b-instruct:free",
            "google/gemma-3-4b-it:free",
        ],
    )
    embed_model = st.selectbox(
        "Embedding model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        help="Sentence-transformers model (runs locally, free)",
    )

    build_btn = st.button("🔨 Build / Rebuild Index", use_container_width=True, type="primary")

# ── Session state ─────────────────────────────────────────────────────────────
if "pipeline"      not in st.session_state: st.session_state.pipeline      = None
if "messages"      not in st.session_state: st.session_state.messages      = []
if "index_ready"   not in st.session_state: st.session_state.index_ready   = False
if "index_stats"   not in st.session_state: st.session_state.index_stats   = {}

# ── Build index ───────────────────────────────────────────────────────────────
if build_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one document first.")
    else:
        with st.spinner("Chunking & embedding documents…"):
            try:
                pipeline = RAGPipeline(
                    openrouter_api_key=openrouter_key or None,
                    model_name=model_choice,
                    embed_model=embed_model,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                )
                stats = pipeline.build_index(uploaded_files)
                st.session_state.pipeline    = pipeline
                st.session_state.index_ready = True
                st.session_state.index_stats = stats
                st.session_state.messages    = []
                st.sidebar.success(f"✅ Index built — {stats['n_chunks']} chunks from {stats['n_docs']} doc(s)")
            except Exception as e:
                st.sidebar.error(f"Index build failed: {e}")

# ── Index stats ───────────────────────────────────────────────────────────────
if st.session_state.index_ready and st.session_state.index_stats:
    s = st.session_state.index_stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents", s.get("n_docs", "-"))
    c2.metric("Chunks",    s.get("n_chunks", "-"))
    c3.metric("Embed dim", s.get("embed_dim", "-"))

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "contexts" in msg:
            with st.expander(f"📚 Retrieved context ({len(msg['contexts'])} chunks)", expanded=False):
                for i, ctx in enumerate(msg["contexts"], 1):
                    st.markdown(f"""
<div class="context-card">
  <div class="chunk-meta">Chunk {i} &nbsp;·&nbsp; score: {ctx['score']:.3f} &nbsp;·&nbsp; source: {ctx['source']}</div>
  {ctx['text']}
</div>""", unsafe_allow_html=True)
            st.markdown(f"""
<div class="metric-row">
  <span class="metric-pill">⏱ {msg.get('latency','?')}s</span>
  <span class="metric-pill">🔍 {len(msg.get('contexts',[]))} chunks used</span>
  <span class="metric-pill">🤖 {msg.get('model','?').split('/')[-1]}</span>
</div>""", unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
example_queries = [
    "What factors affect construction project delays?",
    "What are the safety requirements on site?",
    "How are contractor payments processed?",
]

if not st.session_state.index_ready:
    st.info("👈 Upload documents and click **Build / Rebuild Index** to start chatting.")
    st.subheader("Example questions you'll be able to ask:")
    for q in example_queries:
        st.markdown(f"- *{q}*")
else:
    user_input = st.chat_input("Ask a question about your documents…")

    # Quick example buttons
    cols = st.columns(len(example_queries))
    for col, q in zip(cols, example_queries):
        if col.button(q, use_container_width=True):
            user_input = q

    if user_input:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Retrieving & generating…"):
                try:
                    t0     = time.time()
                    result = st.session_state.pipeline.query(user_input, top_k=top_k)
                    latency = f"{time.time() - t0:.1f}"

                    st.markdown(result["answer"])

                    with st.expander(f"📚 Retrieved context ({len(result['contexts'])} chunks)", expanded=False):
                        for i, ctx in enumerate(result["contexts"], 1):
                            st.markdown(f"""
<div class="context-card">
  <div class="chunk-meta">Chunk {i} &nbsp;·&nbsp; score: {ctx['score']:.3f} &nbsp;·&nbsp; source: {ctx['source']}</div>
  {ctx['text']}
</div>""", unsafe_allow_html=True)

                    st.markdown(f"""
<div class="metric-row">
  <span class="metric-pill">⏱ {latency}s</span>
  <span class="metric-pill">🔍 {len(result['contexts'])} chunks used</span>
  <span class="metric-pill">🤖 {model_choice.split('/')[-1]}</span>
</div>""", unsafe_allow_html=True)

                    st.session_state.messages.append({
                        "role":     "assistant",
                        "content":  result["answer"],
                        "contexts": result["contexts"],
                        "latency":  latency,
                        "model":    model_choice,
                    })
                except Exception as e:
                    err = f"⚠️ Error: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
