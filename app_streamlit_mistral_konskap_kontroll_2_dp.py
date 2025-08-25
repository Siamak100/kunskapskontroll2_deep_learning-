# app_streamlit_mistral RAG (Lokal) Mistral + Ollama
# OBS:
# Jag började först med Google Gemini API (3 nycklar från olika gmail konton… ja, jag vet 😅).
# Tanken var om API1 tar slut den hoppa till API2 sen API3. I praktiken mina tänka var kaos.
# Timeouts, token-limit, och varje gång jag ändrade i koden bröts flödet.
# Jag skrev till och med en hybrid (API först, annars lokalt), men det blev för rörigt.
# Slutsats: 100% lokalt med Ollama + Mistral. Mindre “magiskt”, mer kontroll.
# (Och ja – Nomic-embedding försökte äta 33GB RAM på min 16GB-laptop… )

import os
import subprocess
import numpy as np
import faiss
import fitz  # PyMuPDF (jag testade PyPDF2 först men tappade text ibland )
import streamlit as st
from typing import List
from sentence_transformers import SentenceTransformer

# Streamlit , det funkar och går snabbt att visa upp.
st.set_page_config(page_title="RAG (Lokal) - Mistral + Ollama", layout="wide")
st.title(" RAG (Lokal) - Mistral + Ollama")
st.caption("PDF in → FAISS-index → sammanfattning + frågor/svar. Allt lokalt efter första modellen.")

# ---------------- Sidopanel ----------------
st.sidebar.subheader(" Inställningar (lek med dessa om svaren blir konstiga)")
embed_choice = st.sidebar.selectbox(
    "Inbäddningsmodell",
    ["sentence-transformers/all-MiniLM-L6-v2", "nomic-ai/nomic-embed-text-v1.5"],
    index=0  # MiniLM är “det funkar alltid”-läget
)
auto_build = st.sidebar.checkbox(" Bygg FAISS-index automatiskt efter uppladdning", value=False)
top_k = st.sidebar.slider("k (antal segment som hämtas)", 1, 8, 4)  # 3–5 brukar vara lagom
chunk_size = st.sidebar.slider("Segmentstorlek (ord)", 300, 2000, 1200, step=100)  # 1200 funkar bra för boken
overlap = st.sidebar.slider("Överlappning (ord)", 0, 400, 200, step=50)  # lite överlapp = mindre info-tapp
max_summary_sents = st.sidebar.slider("Antal meningar i sammanfattning", 2, 8, 3)

st.sidebar.markdown("**Ladda modeller (engångs):**")
st.sidebar.code("ollama pull mistral\nollama pull nomic-embed-text", language="bash")

# ---------------- Funktioner ----------------
def run_mistral(prompt: str) -> str:
    """
    Kör lokal Mistral via Ollama. Yes, subprocess. Kunde kört HTTP-API också.
    Om du får tomt svar → kolla ollama service, eller att modellen finns lokalt.
    """
    result = subprocess.run(
        ["ollama", "run", "mistral"],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    out = result.stdout.decode("utf-8", errors="ignore").strip()
    if not out:
        st.warning(result.stderr.decode("utf-8", errors="ignore"))
    return out

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """
    Extrahera ren text ur PDF. Jag svettades lite här i början.
    PyMuPDF (fitz) funkar bäst för mig  guld värt om layouten är “normal”.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        parts = []
        for p in doc:
            parts.append(p.get_text("text"))
    return "\n".join(parts)

def chunk_text_words(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Delar upp text i bitar (chunks). För stor chunk = RAM dör (hej Nomic…),
    för liten = modellen fattar inte sammanhanget. 1200/200 blev “lagom” här.
    """
    words = text.split()
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += step
    return chunks

@st.cache_resource(show_spinner=False)
def load_embedder(model_name: str) -> SentenceTransformer:
    """
    Laddar inbäddningsmodellen. OBS: Nomic kräver trust_remote_code=True.
    Det kändes lite trokigt men annars vägrar den köra. MiniLM har inte det kravet.
    """
    if model_name == "nomic-ai/nomic-embed-text-v1.5":
        return SentenceTransformer(model_name, trust_remote_code=True)
    return SentenceTransformer(model_name)

def encode_texts(embedder: SentenceTransformer, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Skapar embeddings. Om det här segar: sänk batch_size (t.ex. 16 eller 8).
    Jag försöker hålla det enkelt här – ingen GPU-trollkonst, bara rakt på.
    """
    vecs = embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vecs.astype("float32")

def build_faiss_ip(embeddings: np.ndarray) -> faiss.Index:
    """
    Bygger ett FAISS-index (inner product + L2-normalisering).
    Annoy/HNSW är fina, men FAISS är stabilt och “just works” för mig.
    """
    dim = embeddings.shape[1]
    idx = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    idx.add(embeddings)
    return idx

def retrieve(embedder: SentenceTransformer, index: faiss.Index, chunks: List[str], query: str, top_k: int):
    """
    Hämtar de mest relevanta textbitarna för frågan.
    Om du byter embedding-modell behöver du bygga om indexet,
     annars kan det bli fel på dimensionerna.

    """
    q = encode_texts(embedder, [query])
    faiss.normalize_L2(q)
    D, I = index.search(q, top_k)
    hits = [chunks[i] for i in I[0]]
    scores = D[0].tolist()
    return hits, scores

def summarize_context(context: str, max_sents: int) -> str:
    """
    Sammanfattar texten kort och bara med fakta.
    Bra när träffarna är långa och vi vill ge modellen en enklare input.
    """
    prompt = (
        f"Gör en mycket kort och exakt sammanfattning av texten nedan. "
        f"Max {max_sents} meningar. Endast fakta.\n\n"
        f"Text:\n{context}\n\nSammanfattning:"
    )
    return run_mistral(prompt)

def answer_with_context(question: str, context: str, summary: str = "") -> str:
    """
    Slutliga svaret: fråga + (sammanfattning) + källtext → Mistral.
    Om svaret inte finns i texten vill jag hellre att den säger det.
    """
    prompt = (
        "Besvara frågan nedan med hjälp av den givna texten. "
        "Om svaret inte finns i texten, skriv 'Finns ej i texten'.\n\n"
        f"Sammanfattning (frivillig): {summary}\n\n"
        f"Textkälla:\n{context}\n\n"
        f"Fråga: {question}\nSvar:"
    )
    return run_mistral(prompt)

# ---------------- Huvuddel ----------------
uploaded = st.file_uploader("Välj en PDF-fil", type=["pdf"])
if uploaded:
    # Här kände jag mig nervös första gången – stora PDF:er kan ta en stund.
    with st.spinner(" Extraherar text från PDF ..."):
        full_text = pdf_bytes_to_text(uploaded.read())
    with st.spinner(" Delar upp text i segment ..."):
        chunks = chunk_text_words(full_text, chunk_size=chunk_size, overlap=overlap)
    st.success(f" Antal segment: {len(chunks)} (om detta är <50 har du nog för stor chunk-size)")

    # En enkel cache-nyckel så vi inte bygger om hela tiden i onödan.
    cache_key = (hash(full_text), embed_choice, chunk_size, overlap)
    if "stores" not in st.session_state:
        st.session_state["stores"] = {}
    stores = st.session_state["stores"]

    def build_and_cache():
        with st.spinner(" Laddar inbäddningsmodell ..."):
            embedder = load_embedder(embed_choice)
        with st.spinner(" Skapar embeddingar och bygger FAISS-index ... "):
            vecs = encode_texts(embedder, chunks)
            index = build_faiss_ip(vecs)
        stores[cache_key] = {"embedder": embedder, "index": index, "chunks": chunks}
        st.success(" FAISS-index byggt. (Skönt!)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(" Bygg FAISS-index"):
            build_and_cache()
    with col2:
        if auto_build and cache_key not in stores:
            build_and_cache()

    if cache_key in stores:
        store = stores[cache_key]
        st.divider()
        st.subheader(" Frågor och svar (RAG)")

        q = st.text_input("Ställ en fråga:", placeholder="Exempel: Vad är ett konvolutionslager?")
        if q:
            with st.spinner(" Söker relevanta segment ..."):
                hits, scores = retrieve(store["embedder"], store["index"], store["chunks"], q, top_k=top_k)
                context = "\n\n---\n\n".join(hits)
            with st.spinner(" Sammanfattar text ..."):
                brief = summarize_context(context, max_sents=max_summary_sents)
            with st.spinner(" Genererar svar ..."):
                answer = answer_with_context(q, context, summary=brief)

            st.markdown("###  Relevanta segment")
            for i, (h, s) in enumerate(zip(hits, scores), 1):
                with st.expander(f"Segment {i}  |  score={s:.3f}"):
                    st.write(h)

            st.markdown("###  Sammanfattning")
            st.write(brief)

            st.markdown("###  Svar")
            st.write(answer)
else:
    st.info(" Ladda upp en PDF för att börja.")

"""
------------------------------------------------
Kommentarer kring API och lokal lösning (ärlig version)
------------------------------------------------
Jag använde Google Gemini API först (API1, API2, API3). Det funkade
ibland första körningen, men sen: timeouts, token-limits  och jag
blev tokig. Jag skrev en fallback (API → lokalt), men det blev för mycket
“tejpa ihop”-känsla. Så jag bestämde mig: bara lokalt med Ollama + Mistral.
Är det perfekt? ja, och jag kan åtminstone jobba
utan att be en server om lov varje gång.
"""
