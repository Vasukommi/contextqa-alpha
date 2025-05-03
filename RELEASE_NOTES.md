# 🚀 Flynkle AlphaV1.0.0 – First Public Alpha

**Release Date:** 2025-05-01  
**Codename:** “Grounded Knowledge”

This is the first working version of Flynkle’s document-based knowledge assistant. Designed to parse internal company files and answer natural language questions grounded in those documents.

---

## ✅ Core Features

- **🗂 Document Uploads**
  - Supports `.pdf`, `.docx`, and `.txt` formats
  - Drag-and-drop or file browser UI via Streamlit

- **📄 Document Parsing + Chunking**
  - Extracts text from uploaded files
  - Splits content into overlapping semantic chunks

- **🧠 Embedding Generation**
  - Uses `sentence-transformers` (`all-MiniLM-L6-v2`)
  - Embeddings saved dynamically per document

- **🧱 FAISS Indexing**
  - Per-document vector index for retrieval
  - Auto-generated on first use

- **💬 Question Answering**
  - Streamlit UI for natural language queries
  - Queries are matched to best chunks
  - GPT answers using only matched content

- **🌐 Dockerized Deployment**
  - `Dockerfile` + `docker-compose.yml` included
  - All volumes and hot reload enabled

---

## 🐞 Known Limitations

- GPT may still say “I couldn’t find that info” if chunks are ambiguous
- No feedback loop or re-ranking (yet)
- Chunk-level source highlights are basic
- No authentication or multi-user state

---

## 💡 Next Up (Planned for Beta)

- Better prompt tuning and summarization
- Multi-file upload and dynamic switching
- Agent memory / persona support
- Feedback buttons (Was this answer helpful?)
- Admin dashboard for analytics

---

## 👨‍💻 For Developers

- Default port: `http://0.0.0.0:8501`
- Embeddings: saved in `/knowledge/`
- Chunks: saved in `/data/processed/`

---
### Flynkle Alpha v2 — Internal Prototype
This is a stripped-down open prototype of the core Flynkle engine, focused on:

- 💡 Document understanding
- 🧠 Embedding + retrieval (via FAISS)
- 🤖 GPT-powered question answering

> Note: This version is intended for internal R&D and sharing the underlying architecture. Branding and commercial layers are excluded.

---

### Core Features
- Upload PDFs, DOCX, TXT
- Parse, chunk, embed with SentenceTransformer
- Store embeddings in FAISS + query with GPT
- Streamlit frontend for live testing

---

### Next Steps
- Add UI filters, auth, multi-user support
- Explore document-level access control
- Migrate to production-grade frontends (e.g., Next.js, React)

Built with ❤️ using Python, Streamlit, SBERT, FAISS, and OpenAI GPT.