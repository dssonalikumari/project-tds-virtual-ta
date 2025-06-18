# main.py
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows tesseract path
import base64
import json
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

# === CONFIG ===
EMBEDDINGS_PATH = "embeddings_meta.npz"
INDEX_PATH = "faiss.index"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
OPENAI_MODEL = "gpt-3.5-turbo"

# === Load Data ===
data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
metas = data["metas"]
embeddings = data["embeddings"]
texts = data["texts"]
index = faiss.read_index(INDEX_PATH)
model = SentenceTransformer(MODEL_NAME)

# === FastAPI setup ===
app = FastAPI()

# === Request Schema ===
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

def extract_links_from_response(response_text):
    links = []

    def clean(val):
        return val.strip().strip('",)')

    # From Sources: URL: [...], Text: [...]
    pattern = r"URL:\s*\[?(https?://[^\]\s]+)\]?\s*,?\s*Text:\s*\[?(.+?)\]?(?:\n|$)"
    matches = re.findall(pattern, response_text)
    for url, text in matches:
        links.append({"url": clean(url), "text": clean(text)})

    # Markdown links
    inline_links = re.findall(r"\[(.*?)\]\((https?://[^\)]+)\)", response_text)
    for text, url in inline_links:
        links.append({"url": clean(url), "text": clean(text)})

    # Raw URLs
    raw_urls = re.findall(r"(https?://[^\s\]\)\}]+)", response_text)
    for url in raw_urls:
        if not any(l["url"] == clean(url) for l in links):
            links.append({"url": clean(url), "text": clean(url)})

    # ✅ Normalize Discourse URLs to thread-level only
    for link in links:
        if "discourse.onlinedegree.iitm.ac.in" in link["url"]:
            link["url"] = re.sub(r"/\d+$", "", link["url"])

    return links

# === Main API ===
@app.post("/api/")
async def answer_question(req: QueryRequest):
    # --- OCR Step ---
    extracted_text = ""
    if req.image:
        try:
            img_data = base64.b64decode(req.image)
            img = Image.open(BytesIO(img_data))
            extracted_text = pytesseract.image_to_string(img)
        except Exception as e:
            extracted_text = f"[OCR Error: {str(e)}]"

    full_query = req.question + ("\n" + extracted_text if extracted_text else "")

    # --- Embedding ---
    query_embedding = model.encode([full_query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)

    # --- FAISS search ---
    D, I = index.search(query_embedding, k=5)

    context = "\n---\n".join([
        f"[{metas[i]['source']}] {metas[i].get('topic', metas[i].get('path', ''))} -- {i}:\n{texts[i]}"
        for i in I[0]
    ])

    prompt = f"""
You are a helpful assistant for IITM Online Degree program. Use only the CONTEXT to answer the QUESTION.
If the context does not contain enough information to answer the question, say:
"I don't have enough information to answer this question."

Important: When quoting numbers (like scores), use them **exactly as shown** in the context (do not simplify or modify them).


Always include sources in this format:

Sources:
1. URL: [https://example.com/page], Text: [short quote or summary]
2. URL: [https://discourse.onlinedegree.iitm.ac.in/...], Text: [relevant clarification]

---

Context:
{context}

Question:
{req.question}

Your complete answer:
"""

    # --- OpenAI Call ---
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a data science course. Always answer based only on the given context. Provide useful links if available."},
                {"role": "user", "content": prompt}
            ]
        )

        answer = response.choices[0].message.content.strip()

        # Remove ```json or ``` wrappers
        if answer.startswith("```json"):
            answer = answer.replace("```json", "").strip("` \n")
        elif answer.startswith("```"):
            answer = answer.replace("```", "").strip("` \n")

    except Exception as e:
        answer = f"OpenAI API error: {str(e)}"

    # --- Link extraction ---
    links = extract_links_from_response(answer)

    # Fallback: Use metadata if GPT doesn't return sources
    if not links:
        links = []
        for i in I[0]:
            meta = metas[i]
            source = meta.get("source", "unknown")
            url = meta.get("url", "#")
            label = meta.get("topic") or meta.get("title") or meta.get("path", "Course Resource")
            links.append({
                "url": url,
                "text": label
            })

    # Debug Output (Optional)
    print("\n======= GPT RESPONSE =======")
    print(answer)
    print("\n======= EXTRACTED LINKS =======")
    print(links)

    return {
        "answer": answer,
        "question_received": req.question,
        "ocr_text": extracted_text,
        "links": links
    }

# === Health Check ===
@app.get("/health")
def health_check():
    try:
        return {
            "status": "ok",
            "openai_model": OPENAI_MODEL,
            "index_loaded": True,
            "num_chunks": len(texts),
            "vector_dim": VECTOR_DIM
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }
if __name__ == "__main__":
    import uvicorn
    # fallback to 8000 locally
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
