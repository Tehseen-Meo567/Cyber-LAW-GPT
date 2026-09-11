"""
Cyber Law GPT
-------------
A free-to-run Retrieval-Augmented-Generation (RAG) app that answers questions
strictly grounded in a cyber-law PDF document.

Runs on: local machine, Google Colab, Streamlit Community Cloud, or any
generic cloud VM. No paid services are required:
  - Embeddings  -> sentence-transformers (runs locally, no API key)
  - Vector store -> FAISS (in-memory, no external DB)
  - LLM answer  -> Groq API (has a permanent free tier, get a key at
                    https://console.groq.com/keys). If no key is supplied the
                    app still works: it returns the most relevant extracted
                    passages instead of a generated answer.

IMPORTANT LEGAL NOTE
This app answers using content extracted from the PDF you provide. It is an
informational tool, NOT a substitute for advice from a licensed lawyer.
"""

import os
import re
import hashlib
import tempfile

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Cyber Law GPT", page_icon="⚖️", layout="wide")

DEFAULT_DRIVE_LINK = "https://drive.google.com/file/d/1ZMw9Pl-KJ4fAUHBX-EbcKZdsu2SqBrOi/view?usp=drive_link"
CHUNK_WORDS = 350
CHUNK_OVERLAP = 60
TOP_K = 5

# ----------------------------------------------------------------------------
# Helpers: getting the PDF (Google Drive download OR manual upload fallback)
# ----------------------------------------------------------------------------
def _extract_drive_id(url_or_id: str) -> str:
    """Pull the file ID out of a Google Drive share link, or return as-is."""
    match = re.search(r"/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id.strip()


def download_pdf_from_drive(drive_link: str, dest_path: str) -> tuple[bool, str]:
    """Download a PDF from a public Google Drive link using gdown."""
    try:
        import gdown
    except ImportError:
        return False, "gdown is not installed. Add it to requirements.txt."

    file_id = _extract_drive_id(drive_link)
    url = f"https://drive.google.com/uc?id={file_id}"

    def _finish(result_path):
        if result_path and os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            return True, "Downloaded from Google Drive."
        return False, "Download produced an empty file (link may not be public)."

    # Newer gdown versions accept fuzzy=True (lets you pass a full share link).
    # Older versions (installed on some Streamlit Cloud environments) don't
    # know that argument, so fall back to the plain call if it errors out.
    try:
        result_path = gdown.download(url, dest_path, quiet=True, fuzzy=True)
        return _finish(result_path)
    except TypeError:
        pass
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not download from Google Drive: {exc}"

    try:
        result_path = gdown.download(url, dest_path, quiet=True)
        return _finish(result_path)
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not download from Google Drive: {exc}"


@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """Return a list of {'page': int, 'text': str} for every page of the PDF."""
    from pypdf import PdfReader
    import io

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split page text into overlapping word-chunks, keeping page number metadata."""
    chunks = []
    for p in pages:
        words = p["text"].split()
        start = 0
        while start < len(words):
            end = start + CHUNK_WORDS
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({"page": p["page"], "text": chunk_text})
            start = end - CHUNK_OVERLAP
            if end >= len(words):
                break
    return chunks


# ----------------------------------------------------------------------------
# Embeddings + FAISS index (cached so it only runs once at startup)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_index(doc_hash: str, chunks: list[dict]):
    """Build a FAISS cosine-similarity index. Cached per document hash."""
    import faiss

    embedder = load_embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve(query: str, chunks: list[dict], index, top_k: int = TOP_K):
    embedder = load_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    import faiss
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# ----------------------------------------------------------------------------
# Prompting + LLM call (Groq free tier). Falls back to extractive mode.
# ----------------------------------------------------------------------------
TECH_LEVEL_INSTRUCTIONS = {
    "Beginner (plain language)": (
        "Explain in simple, everyday language. Avoid legal jargon; when a "
        "technical or legal term is unavoidable, briefly define it in parentheses."
    ),
    "Intermediate": (
        "Use clear professional language. You may use standard legal/technical "
        "terms but briefly clarify the less common ones."
    ),
    "Expert (legal/technical)": (
        "Respond as you would to a lawyer or cybersecurity professional. Use "
        "precise legal and technical terminology, section numbers, and formal tone."
    ),
}

LENGTH_INSTRUCTIONS = {
    "Short (2-3 sentences)": "Answer in 2-3 concise sentences.",
    "Medium (a paragraph)": "Answer in one well-organized paragraph.",
    "Detailed (in-depth, with sections)": (
        "Give a detailed, structured answer using short headings or bullet "
        "points where useful."
    ),
}

LANGUAGE_INSTRUCTIONS = {
    "English": "Respond in English.",
    "Urdu": "Respond in Urdu (اردو).",
}


def build_prompt(question, context_chunks, tech_level, length, language):
    context_text = "\n\n".join(
        f"[Page {c['page']}] {c['text']}" for c in context_chunks
    )
    system_prompt = (
        "You are Cyber Law GPT, an assistant that answers questions ONLY using "
        "the cyber-law document excerpts provided below. "
        "If the answer is not contained in the excerpts, say clearly that the "
        "document does not cover it — do not invent legal information. "
        "Always remind the user this is informational, not formal legal advice, "
        "if the question seeks specific legal guidance.\n\n"
        f"Style instructions: {TECH_LEVEL_INSTRUCTIONS[tech_level]} "
        f"{LENGTH_INSTRUCTIONS[length]} {LANGUAGE_INSTRUCTIONS[language]}\n\n"
        f"--- DOCUMENT EXCERPTS ---\n{context_text}\n--- END EXCERPTS ---"
    )
    return system_prompt


def call_groq(system_prompt: str, question: str, api_key: str) -> str:
    from groq import Groq

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", "llama-3.1-8b-instant"
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# ----------------------------------------------------------------------------
# Sidebar UI
# ----------------------------------------------------------------------------
st.sidebar.title("⚖️ Cyber Law GPT — Settings")

groq_key = st.sidebar.text_input(
    "Groq API key (free)",
    type="password",
    value=os.environ.get("GROQ_API_KEY", ""),
    help="Get a free key at https://console.groq.com/keys. "
         "Without a key the app still works, but it will show the most "
         "relevant document passages instead of a generated answer.",
)

st.sidebar.markdown("### Response settings")
tech_level = st.sidebar.selectbox(
    "Technical level", list(TECH_LEVEL_INSTRUCTIONS.keys()), index=1
)
resp_length = st.sidebar.selectbox(
    "Response size", list(LENGTH_INSTRUCTIONS.keys()), index=1
)
language = st.sidebar.selectbox(
    "Answer language", list(LANGUAGE_INSTRUCTIONS.keys()), index=0
)
show_sources = st.sidebar.checkbox("Show source passages used", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### Document source")
st.sidebar.caption(
    "Priority: a PDF bundled in the app folder > Google Drive link > manual upload."
)
drive_link = st.sidebar.text_input("Google Drive PDF link (fallback)", value=DEFAULT_DRIVE_LINK)
uploaded_file = st.sidebar.file_uploader(
    "...or upload the PDF manually (used if the other two fail)", type=["pdf"]
)
reload_btn = st.sidebar.button("🔄 Rebuild index from source")

# ----------------------------------------------------------------------------
# Load the document (local file -> Drive download -> manual upload -> embed).
# Runs at startup and is cached, so it only happens once per document.
# ----------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def find_bundled_pdf() -> str | None:
    """Look for a PDF committed alongside app.py in the repo (most reliable option)."""
    import glob

    candidates = sorted(glob.glob(os.path.join(APP_DIR, "*.pdf")))
    return candidates[0] if candidates else None


def get_pdf_bytes():
    # 1) A PDF committed to the repo next to app.py — most reliable, no
    #    network dependency, works identically on every platform.
    bundled_path = find_bundled_pdf()
    if bundled_path:
        with open(bundled_path, "rb") as f:
            return f.read(), f"Loaded bundled file: {os.path.basename(bundled_path)}"

    # 2) Manual upload from the sidebar.
    if uploaded_file is not None:
        return uploaded_file.read(), "Loaded from manual upload."

    # 3) Google Drive download (least reliable — depends on Drive permissions
    #    and Google's automated-download limits).
    tmp_path = os.path.join(tempfile.gettempdir(), "cyberlaw_source.pdf")
    ok, msg = download_pdf_from_drive(drive_link, tmp_path)
    if ok:
        with open(tmp_path, "rb") as f:
            return f.read(), msg
    return None, msg


if "pdf_bytes" not in st.session_state or reload_btn:
    with st.spinner("Fetching document and building embeddings... (first run only)"):
        pdf_bytes, status_msg = get_pdf_bytes()
        st.session_state["pdf_bytes"] = pdf_bytes
        st.session_state["status_msg"] = status_msg

pdf_bytes = st.session_state.get("pdf_bytes")
status_msg = st.session_state.get("status_msg", "")

st.title("⚖️ Cyber Law GPT")
st.caption(
    "Ask questions about the cyber-law document. Answers are grounded strictly "
    "in the document content — this is not a substitute for professional legal advice."
)

if not pdf_bytes:
    st.error(
        f"Could not load a source PDF ({status_msg}). "
        "Either commit a .pdf file into the app's repo folder, fix the Google "
        "Drive link, or upload the PDF using the sidebar."
    )
    st.stop()
else:
    st.sidebar.success(status_msg)

pages = extract_text_from_pdf(pdf_bytes)
if not pages:
    st.error("No extractable text was found in this PDF (it may be a scanned image). "
              "Try a text-based PDF or OCR it first.")
    st.stop()

chunks = chunk_pages(pages)
doc_hash = hashlib.sha256(pdf_bytes).hexdigest()

with st.spinner("Building embeddings index..."):
    index, _ = build_index(doc_hash, chunks)

st.sidebar.info(f"Indexed {len(pages)} pages / {len(chunks)} chunks.")

# ----------------------------------------------------------------------------
# Chat UI
# ----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the cyber law document...")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and preparing answer..."):
            top_chunks = retrieve(question, chunks, index, TOP_K)

            if groq_key:
                try:
                    system_prompt = build_prompt(
                        question, top_chunks, tech_level, resp_length, language
                    )
                    answer = call_groq(system_prompt, question, groq_key)
                except Exception as exc:  # noqa: BLE001
                    answer = (
                        f"⚠️ Could not reach the LLM ({exc}). Showing the most "
                        "relevant passages from the document instead:\n\n"
                        + "\n\n".join(
                            f"**Page {c['page']}:** {c['text'][:600]}..."
                            for c in top_chunks
                        )
                    )
            else:
                answer = (
                    "*(No Groq API key set — showing the most relevant passages "
                    "found in the document. Add a free key in the sidebar to get "
                    "a generated answer instead.)*\n\n"
                    + "\n\n".join(
                        f"**Page {c['page']}:** {c['text'][:600]}..."
                        for c in top_chunks
                    )
                )

            st.markdown(answer)

            if show_sources and groq_key:
                with st.expander("📄 Source passages used"):
                    for c in top_chunks:
                        st.markdown(f"**Page {c['page']}** (relevance {c['score']:.2f})")
                        st.write(c["text"][:800] + ("..." if len(c["text"]) > 800 else ""))

    st.session_state["messages"].append({"role": "assistant", "content": answer})

st.caption(
    "Disclaimer: Cyber Law GPT provides general information drawn from the "
    "supplied document only. It is not legal advice. Consult a qualified "
    "lawyer for advice on a specific situation."
)
