"""Campus Help Desk - a small RAG (retrieval-augmented generation) chatbot.

It answers questions using only the documents in the ./docs folder.

Modes:
- AI mode: set OPENAI_API_KEY. Documents are embedded with OpenAI embeddings,
  the best passages are retrieved, and a chat model writes the answer.
- Demo mode: no API key needed. A built-in BM25 keyword search finds the best
  passages and shows them as the answer. Useful for testing and screenshots.

Run:  uvicorn app:app --reload   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent

# Read simple KEY=VALUE lines from .env (if present) without extra packages.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.lstrip().startswith("#"):
            _key, _value = _line.split("=", 1)
            os.environ.setdefault(_key.strip(), _value.strip())

DOCS_DIR = Path(os.getenv("DOCS_DIR", BASE_DIR / "docs"))
CHUNK_WORDS = 120
CHUNK_OVERLAP = 30
TOP_K = 3

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

STOPWORDS = set(
    "a an and are as at be by can do does for from how i if in is it of on or "
    "my our the to was what when where which who will with you your".split()
)


# ---------------------------------------------------------------------------
# Loading and chunking documents
# ---------------------------------------------------------------------------
@dataclass
class Chunk:
    source: str
    title: str
    text: str
    tokens: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split a markdown file into (heading, body) sections."""
    sections: list[tuple[str, str]] = []
    heading, lines = "", []
    for line in markdown.splitlines():
        if line.startswith("#"):
            if "".join(lines).strip():
                sections.append((heading, "\n".join(lines).strip()))
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    if "".join(lines).strip():
        sections.append((heading, "\n".join(lines).strip()))
    return sections


def chunk_words(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_WORDS:
        return [text]
    step = CHUNK_WORDS - CHUNK_OVERLAP
    return [" ".join(words[i : i + CHUNK_WORDS]) for i in range(0, len(words), step)]


def load_chunks(docs_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        for heading, body in split_sections(path.read_text(encoding="utf-8")):
            for piece in chunk_words(body):
                title = heading or path.stem.replace("-", " ").title()
                chunks.append(Chunk(path.name, title, piece, tokenize(f"{title} {piece}")))
    return chunks


# ---------------------------------------------------------------------------
# Retrieval: BM25 (no API key) or embeddings (with API key)
# ---------------------------------------------------------------------------
class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.avgdl = sum(len(d) for d in docs) / max(len(docs), 1)
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in docs]

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for tf, doc in zip(self.tf, self.docs):
            s = 0.0
            for q in query:
                if q in tf:
                    f = tf[q]
                    s += self.idf[q] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(doc) / self.avgdl)
                    )
            out.append(s)
        return out


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class KnowledgeBase:
    def __init__(self, docs_dir: Path, api_key: str = ""):
        self.chunks = load_chunks(docs_dir)
        if not self.chunks:
            raise RuntimeError(f"No .md or .txt files found in {docs_dir}")
        self.bm25 = BM25([c.tokens for c in self.chunks])
        self.client = None
        if api_key:
            from openai import OpenAI  # imported only when AI mode is used

            self.client = OpenAI(api_key=api_key)
            vectors = self.client.embeddings.create(
                model=EMBED_MODEL, input=[f"{c.title}\n{c.text}" for c in self.chunks]
            ).data
            for chunk, vec in zip(self.chunks, vectors):
                chunk.embedding = vec.embedding

    @property
    def mode(self) -> str:
        return "ai" if self.client else "demo"

    def search(self, question: str, k: int = TOP_K) -> list[tuple[Chunk, float]]:
        if self.client:
            q = self.client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
            scored = [(c, cosine(q, c.embedding or [])) for c in self.chunks]
        else:
            scored = list(zip(self.chunks, self.bm25.scores(tokenize(question))))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [(c, s) for c, s in scored[:k] if s > 0]

    def answer(self, question: str) -> dict:
        hits = self.search(question)
        sources = [
            {"file": c.source, "section": c.title, "text": c.text, "score": round(s, 3)}
            for c, s in hits
        ]
        if not hits:
            return {
                "mode": self.mode,
                "answer": "I couldn't find this in the documents. Please contact the help desk.",
                "sources": [],
            }
        if not self.client:
            best = hits[0][0]
            return {
                "mode": "demo",
                "answer": f"From \"{best.title}\" ({best.source}):\n\n{best.text}",
                "sources": sources,
            }
        context = "\n\n".join(f"[{i + 1}] {c.title} ({c.source})\n{c.text}" for i, (c, _) in enumerate(hits))
        reply = self.client.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful campus help desk assistant. Answer ONLY from the "
                        "numbered context. Cite sources like [1]. If the answer is not in the "
                        "context, say you don't know and suggest contacting the help desk."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return {"mode": "ai", "answer": reply.choices[0].message.content, "sources": sources}


# ---------------------------------------------------------------------------
# Web app
# ---------------------------------------------------------------------------
app = FastAPI(title="Campus Help Desk Chatbot")
kb = KnowledgeBase(DOCS_DIR, OPENAI_API_KEY)


class Question(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "mode": kb.mode, "chunks": len(kb.chunks)}


@app.post("/api/ask")
def ask(q: Question) -> dict:
    try:
        return kb.answer(q.question.strip())
    except Exception as exc:  # show a readable error instead of a crash
        raise HTTPException(status_code=502, detail=f"AI service error: {exc}") from exc


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "static" / "index.html")
