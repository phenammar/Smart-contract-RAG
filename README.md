# 📄 Smart Contract Q&A Assistant

A simple **Retrieval-Augmented Generation (RAG)** web application built with:

* **LangChain**
* **Chroma Vector Database**
* **HuggingFace Embeddings**
* **Groq LLaMA 3.1**
* **Gradio UI**

The app allows users to upload a PDF document and ask questions about its content.

---

## 🚀 Features

* 📂 Upload any PDF file
* ✂ Automatically split document into chunks
* 🧠 Generate embeddings using `all-MiniLM-L6-v2`
* 📦 Store vectors in Chroma DB
* 🔎 Perform semantic search
* 🤖 Answer questions using Groq LLaMA 3.1
* 🌐 Simple Gradio web interface

---

## 🧠 How It Works

1. Upload a PDF
2. Document is split into chunks
3. Chunks are embedded into vectors
4. Stored in Chroma vector database
5. User asks a question
6. Top relevant chunks are retrieved
7. LLM generates answer using retrieved context

---

## 📁 Project Structure

```
app.py               # (main app)
chroma_dynamic/      # Vector DB

create_database.py   # (offline testing)
query.py             # (testing script)
chroma_db/           # Vector DB (for the testing files)
data/docs            # Contains a sample pdf for testing

```

---

## ⚙ Installation

```bash
pip install -r requirements.txt
```


---


## ▶ Run The App

```bash
python app.py
```

The app will launch locally and optionally create a public Gradio share link.

---


## 📌 Tech Stack

* Python
* LangChain
* ChromaDB
* HuggingFace Sentence Transformers
* Groq LLaMA 3.1
* Gradio

---



## 👨‍💻 Author

Built as a learning project for mastering RAG systems.
