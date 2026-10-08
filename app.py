"""Simple ChatPDF: upload a PDF, ask questions about it.
Run:  python -m pip install -r requirements.txt  ->  fill .env  ->  python app.py
Open: http://127.0.0.1:5000
"""
import os
import re
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL") or None,
    timeout=60,
)
MODEL = os.getenv("LLM_MODEL", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB
DOCS: dict[str, list[str]] = {}  # doc_id -> text chunks (in memory only)


def split_chunks(text: str, size: int = 1000, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def top_chunks(chunks: list[str], question: str, k: int = 4) -> list[str]:
    q = words(question)
    scored = [(len(q & words(c)), i) for i, c in enumerate(chunks)]
    if max(s for s, _ in scored) == 0:
        # Generic question ("summarize this"): use the beginning of the document.
        return chunks[:k]
    best = sorted(scored, key=lambda s: (-s[0], s[1]))[:k]
    return [chunks[i] for _, i in sorted(best, key=lambda s: s[1])]  # keep reading order


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename.lower().endswith(".pdf"):
        return jsonify(error="That is not a PDF. Choose a file ending in .pdf."), 400
    try:
        reader = PdfReader(file)
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    except Exception:
        return jsonify(error="This PDF could not be read. It may be damaged or password-protected."), 400
    if not text:
        return jsonify(error="No text found. This looks like a scanned PDF, which needs OCR."), 400
    doc_id = uuid.uuid4().hex
    DOCS[doc_id] = split_chunks(text)
    return jsonify(doc_id=doc_id, filename=file.filename, pages=len(reader.pages), chars=len(text))


@app.post("/chat")
def chat():
    data = request.get_json(force=True)
    chunks = DOCS.get(data.get("doc_id"))
    question = (data.get("question") or "").strip()
    if chunks is None or not question:
        return jsonify(error="Upload a PDF first, then type a question."), 400
    picked = top_chunks(chunks, question)
    context = "\n---\n".join(picked)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Answer ONLY from the document excerpts below, "
                 "in the user's language. If the answer is not there, say you can't find it.\n\n"
                 f"EXCERPTS:\n{context}"},
                {"role": "user", "content": question},
            ],
        )
        return jsonify(answer=resp.choices[0].message.content, sources=picked)
    except Exception as exc:
        return jsonify(error=f"The AI service returned an error: {exc}"), 500


if __name__ == "__main__":
    app.run(debug=True)