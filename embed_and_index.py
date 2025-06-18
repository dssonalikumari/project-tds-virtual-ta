# embed_and_index.py
import json
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import re

# ===== CONFIG =====
POSTS_PATH = "tds_posts_with_ocr.json"
COURSE_DIR = "tools-in-data-science-public"
CHUNK_SIZE = 1000  # characters per chunk
OVERLAP = 200
VECTOR_DIM = 384  # for all-MiniLM-L6-v2
INDEX_PATH = "faiss.index"
NPZ_PATH = "embeddings_meta.npz"

# ===== Helper: load markdown files =====
def load_markdown_files(folder_path):
    docs = []
    for md_file in glob.glob(f"{folder_path}/**/*.md", recursive=True):
        with open(md_file, encoding="utf-8") as f:
            text = f.read()
        docs.append({"text": text, "path": md_file})
    return docs

# ===== Helper: chunk text =====
def add_chunks(text, meta_base, texts, metas):
    start = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE]
        texts.append(chunk)
        metas.append(meta_base.copy())
        start += CHUNK_SIZE - OVERLAP

def main():
    # 1. Load datasets
    posts = json.load(open(POSTS_PATH, encoding="utf-8"))
    course_docs = load_markdown_files(COURSE_DIR)
    print(f"Loaded {len(posts)} posts and {len(course_docs)} markdown files")

    # 2. Build chunks and metadata
    texts, metas = [], []
    for p in posts:
        text = p.get("content", "").strip()
        if not text:
            continue

        slug = p.get("slug", "")
        topic_id = p.get("topic_id")
        post_number = p.get("post_number", 1)  # use post_number from post
        meta = {
            "source": "discourse",
            "topic": p.get("topic_title", ""),
            "post_id": p.get("post_id"),
            "topic_id": topic_id,
            "slug": slug,
            "post_number": post_number,
            "url": f"https://discourse.onlinedegree.iitm.ac.in/t/{slug}/{topic_id}/{post_number}"
        }
        add_chunks(text, meta, texts, metas)

    for doc in course_docs:
        text = doc["text"].strip()
        if not text:
            continue
        meta = {
            "source": "course",
            "path": doc.get("path"),
            "title": "",
            "url": ""
        }

        # Try to extract metadata from frontmatter
        with open(doc.get("path"), encoding="utf-8") as f:
            content = f.read()

        frontmatter = re.search(r"---(.*?)---", content, re.DOTALL)
        if frontmatter:
            meta_block = frontmatter.group(1)
            title = re.search(r'title:\s*"(.*?)"', meta_block)
            url = re.search(r'original_url:\s*"(.*?)"', meta_block)
            if title:
                meta["title"] = title.group(1)
            if url:
                meta["url"] = url.group(1)

        add_chunks(text, meta, texts, metas)

    print(f"Created {len(texts)} text chunks")

    # 3. Embed chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    # 4. Build FAISS index
    index = faiss.IndexFlatIP(VECTOR_DIM)
    index.add(embeddings)

    # 5. Save index and metadata
    faiss.write_index(index, INDEX_PATH)
    # Save embeddings + metadata + texts
    np.savez(NPZ_PATH, embeddings=embeddings, metas=np.array(metas, dtype=object), texts=np.array(texts, dtype=object))
    print(f"Saved {len(texts)} chunks and FAISS index.")
    print(f"✅ Saved FAISS index ({INDEX_PATH}) and metadata archive ({NPZ_PATH})")

if __name__ == "__main__":
    main()
