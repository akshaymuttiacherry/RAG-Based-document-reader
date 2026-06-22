import streamlit as st
import numpy as np
import faiss
import tempfile
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from PyPDF2 import PdfReader
import easyocr
import ollama

# ── Models (cached so they load once) ──────────────────────────────────────────
@st.cache_resource
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"])

# ── Text extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(file) -> str:
    reader = PdfReader(file)
    return "".join(
        page.extract_text() or "" for page in reader.pages
    )

def extract_image_text(path: str) -> str:
    reader = load_ocr()
    result = reader.readtext(path)
    return " ".join(item[1] for item in result)

# ── Chunking ────────────────────────────────────────────────────────────────────
def create_chunks(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=100
    )
    return splitter.split_text(text)

# ── Vector DB ───────────────────────────────────────────────────────────────────
class VectorDB:
    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        self.texts: list[str] = []

    def add(self, embeddings, chunks: list[str]):
        self.index.add(np.array(embeddings).astype("float32"))
        self.texts.extend(chunks)

    def search(self, query_embedding, k: int = 5) -> list[str]:
        k = min(k, len(self.texts))
        if k == 0:
            return []
        _, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), k
        )
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]

# ── Hybrid search ───────────────────────────────────────────────────────────────
class HybridSearch:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        tokenized = [c.split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def keyword_search(self, query: str, k: int = 5) -> list[str]:
        scores = self.bm25.get_scores(query.split())
        top_indices = scores.argsort()[::-1][:k]
        return [self.chunks[i] for i in top_indices]

# ── Reranking ───────────────────────────────────────────────────────────────────
def rerank(query: str, docs: list[str]) -> list[str]:
    if not docs:
        return []
    reranker = load_reranker()
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

# ── LLM answer ──────────────────────────────────────────────────────────────────
def generate_answer(context: str, question: str, history: list[dict]) -> str:
    history_text = ""
    if history:
        history_text = "\n".join(
            f"Q: {h['question']}\nA: {h['answer']}" for h in history[-4:]
        )
        history_text = f"\nConversation history:\n{history_text}\n"

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided.
If the answer isn't in the context, say "I don't have enough information to answer that."{history_text}

Context:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

# ── Build index from uploaded files ────────────────────────────────────────────
def build_index(uploaded_files) -> tuple[VectorDB, HybridSearch, list[str]]:
    all_chunks: list[str] = []
    embedder = load_embedder()

    for f in uploaded_files:
        if f.type == "application/pdf":
            text = extract_pdf_text(f)
        else:  # image
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(f.name)[1]) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name
            text = extract_image_text(tmp_path)
            os.unlink(tmp_path)

        chunks = create_chunks(text)
        all_chunks.extend(chunks)

    if not all_chunks:
        return None, None, []

    embeddings = embedder.encode(all_chunks, show_progress_bar=False)
    dimension = embeddings.shape[1]

    vdb = VectorDB(dimension)
    vdb.add(embeddings, all_chunks)

    hybrid = HybridSearch(all_chunks)
    return vdb, hybrid, all_chunks

# ── Streamlit UI ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Agentic RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Agentic RAG Document Assistant")

# Session state initialisation
for key, default in [
    ("vdb", None),
    ("hybrid", None),
    ("chat_history", []),
    ("index_built", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar: upload & index ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Build index", type="primary"):
        with st.spinner("Extracting text and building index…"):
            vdb, hybrid, chunks = build_index(uploaded_files)
            if vdb:
                st.session_state.vdb = vdb
                st.session_state.hybrid = hybrid
                st.session_state.index_built = True
                st.session_state.chat_history = []
                st.success(f"Indexed {len(chunks)} chunks from {len(uploaded_files)} file(s).")
            else:
                st.error("No text could be extracted from the uploaded files.")

    if st.session_state.index_built:
        st.divider()
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()

# ── Main: chat ──────────────────────────────────────────────────────────────────
if not st.session_state.index_built:
    st.info("Upload one or more PDFs or images in the sidebar and click **Build index** to get started.")
else:
    # Render previous messages
    for msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            with st.expander("Sources used"):
                for i, src in enumerate(msg.get("sources", []), 1):
                    st.caption(f"**[{i}]** {src[:300]}{'…' if len(src) > 300 else ''}")

    # New question
    question = st.chat_input("Ask a question about your documents…")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating…"):
                embedder = load_embedder()
                q_emb = embedder.encode(question)

                # Hybrid retrieval
                semantic_hits = st.session_state.vdb.search(q_emb, k=5)
                keyword_hits = st.session_state.hybrid.keyword_search(question, k=5)

                # Deduplicate
                combined = list(dict.fromkeys(semantic_hits + keyword_hits))

                # Rerank
                ranked = rerank(question, combined)
                top_docs = ranked[:5]
                context = "\n\n---\n\n".join(top_docs)

                # Generate
                answer = generate_answer(
                    context, question, st.session_state.chat_history
                )

            st.write(answer)
            with st.expander("Sources used"):
                for i, src in enumerate(top_docs, 1):
                    st.caption(f"**[{i}]** {src[:300]}{'…' if len(src) > 300 else ''}")

        # Persist to history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": top_docs,
        })