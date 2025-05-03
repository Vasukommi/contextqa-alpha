# 🤖 Flynkle Alpha v2 — Document-Based AI Assistant (RAG Demo)

**Flynkle Alpha v2** is an experimental prototype of a document-questioning assistant powered by Retrieval-Augmented Generation (RAG). This project is part of an early alpha series exploring how AI agents can reason over enterprise knowledge bases.

> 🧪 This is not the production version of [Flynkle](https://flynkle.com) — it's a public testbed to explore, iterate, and share the technical journey behind building AI-native tools.

---

## 🚀 What This Project Does

- 📄 **Multi-document upload**: Upload PDFs, DOCX, or TXT files
- 🧠 **Document chunking & embedding**: Uses `sentence-transformers` + FAISS
- 🔍 **Semantic search**: Vector-based top-k retrieval
- 💬 **Question-answering over documents**: Uses OpenAI's GPT with custom instructions
- 🧰 **Fully local vector store**: Powered by FAISS (offline-ready architecture)
- 📦 **Modular backend design**: Built with extensibility and clean separation

---

## 🛠️ Stack

- **Python 3.10**
- **Streamlit** for the frontend
- **FAISS** for vector search
- **sentence-transformers** for embedding
- **OpenAI GPT API** for LLM-based answers

---

## ⚠️ Notes

- This version is meant for demo, exploration, and resume showcasing purposes.
- It’s a stripped-down version of a larger internal product — built quickly, but thoughtfully.
- Source files are organized to make onboarding and extension easier.

---

## 📁 Structure

┣ 📁src
┃ ┣ 📄agent.py
┃ ┣ 📄vector_store.py
┃ ┣ 📄embedder.py
┃ ┣ 📄retriever.py
┃ ┣ 📄config.py
┃ ┗ ...
┣ 📄app.py # Streamlit interface
┣ 📄requirements.txt
┣ 📄README.md
┗ 📄.gitignore


---

## 🧑‍💻 About the Author

This is a side project by a software engineer actively building [Flynkle](https://flynkle.com) — an AI-native platform for compliance, customer intelligence, and internal tooling automation.  

---

## 🧭 Next Steps

This project may evolve into:

- LangChain / LlamaIndex integrations
- Open source connectors (GDrive, Slack, Notion)
- Local LLM fallback for airgapped environments

---

> 🌱 Thanks for checking this out! Feedback, ideas, and collaborations are welcome.
