from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 120


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def extract_pdf(file_bytes: bytes, source: str) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((page_number, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], source: str, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    words_per_chunk = max(80, int(chunk_tokens * 0.75))
    overlap_words = min(words_per_chunk - 1, int(overlap_tokens * 0.75))
    chunks: list[Chunk] = []
    for page_number, text in pages:
        words = text.split()
        start = 0
        while start < len(words):
            end = min(start + words_per_chunk, len(words))
            chunk_text = " ".join(words[start:end]).strip()
            if chunk_text:
                chunks.append(Chunk(chunk_text, source, page_number))
            if end == len(words):
                break
            start = end - overlap_words
    return chunks


def read_corpus(uploaded_files: list, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for uploaded_file in uploaded_files:
        pages = extract_pdf(uploaded_file.getvalue(), uploaded_file.name)
        chunks.extend(chunk_pages(pages, uploaded_file.name, chunk_tokens, overlap_tokens))
    return chunks


def cache_data(function):
    if st.runtime.exists():
        return st.cache_data(show_spinner=False)(function)
    return function


@cache_data
def read_local_corpus(paths: tuple[str, ...], chunk_tokens: int, overlap_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path_string in paths:
        path = Path(path_string)
        pages = extract_pdf(path.read_bytes(), path.name)
        chunks.extend(chunk_pages(pages, path.name, chunk_tokens, overlap_tokens))
    return chunks


class LocalRAG:
    def __init__(self, chunk_size: int = DEFAULT_CHUNK_TOKENS, overlap: int = DEFAULT_OVERLAP_TOKENS):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []

    def build(self) -> int:
        local_paths = tuple(str(path) for path in sorted(ROOT.glob("*.pdf")))
        self.chunks = read_local_corpus(local_paths, self.chunk_size, self.overlap)
        return len(self.chunks)

    def search(self, question: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        return retrieve(question, self.chunks, top_k)


def generate_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    return answer_question(question, results)


def retrieve(query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]:
    texts = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [(chunks[index], float(scores[index])) for index in ranked if scores[index] > 0]


def limit_to_one_sentence(answer: str) -> str:
    normalized = re.sub(r"\s+", " ", answer).strip()
    boundary = re.search(r"[.!?](?:[\"')\]]*)(?=\s|$)", normalized)
    return normalized[:boundary.end()].strip() if boundary else normalized


def build_prompt(question: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{index}] {chunk.source}, page {chunk.page}\n{chunk.text}"
        for index, (chunk, _) in enumerate(retrieved_chunks, start=1)
    )
    return f"""You are a careful research assistant. Answer the question using only the supplied context.
If the context does not contain enough evidence, say that clearly. Do not invent facts or citations.
Answer the question directly, then briefly explain the evidence in exactly one sentence. Cite it with [1] or [2], and never repeat the question or use bullets.

Question: {question}

Context:
{context}
"""


def answer_count_question(retrieved_chunks: list[tuple[Chunk, float]]) -> str | None:
    count_pattern = re.compile(
        r"\b(?P<count>\d[\d,]*)\s+(?P<group>(?:Claude\s+)?(?:people|users|respondents|participants|employees|workers))\b",
        re.IGNORECASE,
    )
    for index, (chunk, _) in enumerate(retrieved_chunks, start=1):
        match = count_pattern.search(chunk.text)
        if match:
            return f"The survey included {match.group('count')} {match.group('group')}, according to the report [{index}]."
    return None


def repeats_question(answer: str, question: str) -> bool:
    normalized_answer = " ".join(re.findall(r"[a-z0-9]+", answer.casefold()))
    normalized_question = " ".join(re.findall(r"[a-z0-9]+", question.casefold()))
    return normalized_answer == normalized_question or normalized_answer.startswith(normalized_question + " ")


def answer_question(question: str, retrieved_chunks: list[tuple[Chunk, float]]) -> str:
    prompt = build_prompt(question, retrieved_chunks)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                input=prompt,
                temperature=0.1,
            )
            model_answer = response.output_text.strip()
            if repeats_question(model_answer, question):
                if re.search(r"\bhow many\b", question, re.IGNORECASE):
                    extracted_answer = answer_count_question(retrieved_chunks)
                    if extracted_answer:
                        return extracted_answer
                return "The retrieved passages do not provide a direct answer to that question."
            return limit_to_one_sentence(model_answer) or "I could not generate an answer from the available evidence."
        except Exception:  # noqa: BLE001 - fall back when the optional model call fails
            st.warning("The language model was unavailable; showing a grounded fallback response.")

    if not retrieved_chunks:
        return "I could not find a relevant passage in the indexed PDFs."
    if re.search(r"\bhow many\b", question, re.IGNORECASE):
        extracted_answer = answer_count_question(retrieved_chunks)
        if extracted_answer:
            return extracted_answer
    sources = "; ".join(
        f"[{index}] {chunk.source}, page {chunk.page}"
        for index, (chunk, _) in enumerate(retrieved_chunks, start=1)
    )
    return f"Relevant passages are available at {sources}; set OPENAI_API_KEY to enable a synthesized answer."


def main() -> None:
    st.set_page_config(page_title="Field Notes RAG", page_icon="⌁", layout="wide")
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
        :root { --ink:#e8eee9; --muted:#a8b4ad; --paper:#101613; --lime:#d7f36b; --line:#334039; --orange:#f0a06f; }
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; color: var(--ink); }
        .stApp { background: radial-gradient(circle at 90% 0%, #29351f 0, transparent 28%), var(--paper); }
        .hero { padding: 2.2rem 0 1.4rem; border-bottom: 1px solid var(--line); }
        .kicker { font: 500 .75rem 'DM Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--orange); }
        h1 { font-size: clamp(2.2rem, 5vw, 4.5rem); line-height: .98; letter-spacing: -.06em; margin: .45rem 0 .8rem; max-width: 760px; }
        .lede { color: var(--muted); max-width: 640px; font-size: 1.05rem; }
        .stat { background: #18211d; border: 1px solid var(--line); padding: 1rem; min-height: 100px; }
        .stat strong { display:block; font-size:1.7rem; letter-spacing:-.05em; }
        .stat span { color:var(--muted); font-size:.8rem; }
        .source { border-left: 3px solid var(--lime); padding: .65rem .85rem; background: rgba(255,255,255,.06); margin: .45rem 0; }
        .source small { color:var(--muted); font-family:'DM Mono',monospace; }
        [data-testid="stSidebar"] { background:#14241f; }
        [data-testid="stSidebar"] * { color:#f6f4ee; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<section class="hero"><div class="kicker">MIS 7623 / retrieval lab</div><h1>Ask the reports<br/>what they know.</h1><p class="lede">A local RAG workbench for grounded answers across your PDF corpus. Retrieve first, then let the prompt do the synthesis.</p></section>',
        unsafe_allow_html=True,
    )

    local_paths = tuple(str(path) for path in sorted(ROOT.glob("*.pdf")))
    with st.sidebar:
        st.markdown("## Index controls")
        chunk_tokens = st.slider("Chunk size (tokens)", 400, 1200, DEFAULT_CHUNK_TOKENS, 50)
        overlap_tokens = st.slider("Overlap (tokens)", 0, 240, DEFAULT_OVERLAP_TOKENS, 20)
        top_k = st.slider("Retrieved passages", 2, 8, 4)
        uploaded = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)
        st.caption("Default: 800 tokens with 120-token overlap. This suits report prose; adjust downward for tables or sharply topic-specific sections.")

    if uploaded:
        chunks = read_corpus(uploaded, chunk_tokens, overlap_tokens)
    else:
        chunks = read_local_corpus(local_paths, chunk_tokens, overlap_tokens) if local_paths else []

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat"><strong>{len(local_paths) + len(uploaded or [])}</strong><span>PDF sources loaded</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat"><strong>{len(chunks):,}</strong><span>retrieval chunks</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="stat"><strong>{chunk_tokens}</strong><span>target tokens / chunk</span></div>', unsafe_allow_html=True)

    st.markdown("### Query the corpus")
    question = st.chat_input("What does the corpus say about the impact of AI on software work?")
    if question:
        if not chunks:
            st.error("No PDF text is available; add a PDF or place reports beside app.py.")
        else:
            with st.chat_message("user"):
                st.write(question)
            matches = retrieve(question, chunks, top_k)
            with st.chat_message("assistant"):
                if matches:
                    st.write(answer_question(question, matches))
                    with st.expander("Retrieved passages"):
                        for index, (chunk, score) in enumerate(matches, start=1):
                            st.markdown(f'<div class="source"><small>[{index}] {chunk.source} / page {chunk.page} / score {score:.3f}</small><br>{chunk.text}</div>', unsafe_allow_html=True)
                else:
                    st.write("I could not find a relevant passage in the indexed PDFs.")

    with st.expander("How the prompt works"):
        st.code(build_prompt("your question", [(chunks[0], 1.0)]) if chunks else "Load a PDF to inspect the prompt.")


if __name__ == "__main__":
    if st.runtime.exists():
        main()
    else:
        raise SystemExit("Run this app with: streamlit run app.py")