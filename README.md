# TDS Virtual TA (Teaching Assistant)

This is a virtual assistant for IITM's **Tools in Data Science (TDS)** course. It can:

- 🔍 Search and retrieve answers from Discourse forum posts and markdown course notes
- 🧠 Use Retrieval-Augmented Generation (RAG) with OpenAI’s GPT models
- 📷 Process images using OCR to extract embedded text from Discourse posts
- 🚀 Serve answers via an HTTP API using FastAPI
- 📊 Evaluate accuracy and reliability using Promptfoo

---

## 🗂 Project Structure

```bash
.
├── main.py                  # FastAPI app (serves answers)
├── embed_and_index.py       # Converts scraped/markdown data into FAISS index
├── discourse_scraper.py     # Scrapes Discourse + performs OCR on images
├── embeddings_meta.npz      # Metadata + chunked text
├── faiss.index              # FAISS vector index for retrieval
├── tds_posts_with_ocr.json  # Final JSON output of Discourse scrape
├── tools-in-data-science-public/
│   └── *.md                 # Official markdown course notes
├── .env                     # Contains OPENAI_API_KEY
├── requirements.txt         # Python package dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

### 🔁 1. Clone and set up virtual environment

```bash
git clone <your-repo-url>
cd tds-virtual-ta

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### 🧪 2. Install Playwright dependencies

After installing the `playwright` Python package, run this once to download browser binaries:

```bash
playwright install
```

---

### 🔑 3. Set your OpenAI API Key

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXX
```

---

### 🌐 4. Scrape Discourse Data (first-time login required)

```bash
python discourse_scraper.py
```

It will open a Chromium browser window. Log into Discourse manually and **click Resume**. The login session will be saved to `auth.json`.

---

### 🧱 5. Embed and Index Content

This will:

- Chunk Discourse + course content
- Generate embeddings
- Save `faiss.index` and `embeddings_meta.npz`

```bash
python embed_and_index.py
```

---

### 🚀 6. Start the FastAPI server

```bash
uvicorn main:app --reload
```

Test with curl:

```bash
curl http://localhost:8000/api/ \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Should I use Podman or Docker for this course?\"}"
```

You’ll receive a JSON response with the answer, OCR (if any), and source links.

---

### 📊 7. (Optional) Evaluate using Promptfoo

If you’re using [Promptfoo](https://promptfoo.dev), run:

```bash
npx promptfoo eval
```

Make sure your `promptfooconfig.yaml` is correctly pointed to `http://localhost:8000/api/`.

---

## 🧠 Dependencies

See `requirements.txt`. Core libraries:

- `fastapi`, `uvicorn`
- `sentence-transformers`, `faiss-cpu`
- `pytesseract`, `Pillow`
- `playwright`, `beautifulsoup4`, `requests`
- `openai`, `python-dotenv`

System requirements:

- Python 3.9+
- Tesseract OCR (must be installed on system)
- Chromium (auto-installed via Playwright)

---

## 🧾 License

MIT License. Built for educational use in IITM’s online degree program.
