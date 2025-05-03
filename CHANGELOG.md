# 📓 CHANGELOG

All notable changes to Flynkle will be documented in this file.

This project follows [Semantic Versioning](https://semver.org/).

---

## [AlphaV1.0.0] – 2025-05-01

🎉 First alpha release of Flynkle’s document-based assistant.

### Added
- File uploader with support for `.pdf`, `.docx`, `.txt`
- Text parsing and chunking logic
- Dynamic SBERT embedding generation (per file)
- FAISS index creation for semantic search
- Streamlit UI for user queries
- GPT-powered answers based on document context
- Docker support with `Dockerfile` and `docker-compose.yml`

### Known Limitations
- No persistent chat history or feedback tracking
- Limited error handling for edge-case files
- FAISS index created lazily on query (not immediately after embedding)

---

> Upcoming: Alpha patch release (`AlphaV1.1.0`) will improve FAISS indexing, filename handling, and add basic tests.

# Changelog

## [Alpha v2] - 2025-05-03
### Added
- Multi-document upload and processing pipeline
- Centralized vector store using FAISS with chunk alignment checks
- OpenAI GPT-based answering with document-context injection
- Streamlit UI with:
  - Upload section
  - Available document list
  - Ask-your-documents interface

### Improvements
- Full error handling across embedding, chunking, and vector loading
- Singleton pattern for `VectorStore` and OpenAI client
- Clean logging system for debugging and traceability

### Known Limitations
- No auth or file deletion yet
- Only basic document metadata visible
- Only L2 search, no re-ranking