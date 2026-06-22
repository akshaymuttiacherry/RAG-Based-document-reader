# RAG-Based-document-reader
Used for extracting answers from PDF
# 📚 Agentic RAG Document Assistant

An advanced Retrieval-Augmented Generation (RAG) application that enables users to chat with PDFs and scanned documents using Llama 3, Hybrid Search, OCR, FAISS Vector Search, and Cross-Encoder Reranking.

## Features

* Multi-document PDF upload
* OCR support for scanned images
* FAISS semantic search
* BM25 keyword search
* Hybrid retrieval pipeline
* Cross-Encoder reranking
* Conversational memory
* Llama 3 via Ollama
* Source citation display
* Streamlit interface

## Tech Stack

* Python
* Streamlit
* Ollama
* Llama 3
* FAISS
* Sentence Transformers
* BM25
* EasyOCR
* PyPDF2

## Architecture

1. Upload PDFs or Images
2. Extract text
3. Split into chunks
4. Generate embeddings
5. Store vectors in FAISS
6. Perform Hybrid Retrieval
7. Rerank results
8. Generate answer using Llama 3
9. Display sources

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Document-Assistant.git

cd RAG-Document-Assistant

pip install -r requirements.txt
```

Install Ollama and pull Llama 3:

```bash
ollama pull llama3
```

Run:

```bash
streamlit run app.py
```

## Screenshots

Add screenshots inside the assets folder and update links.

## Future Improvements

* LangGraph Agent Workflow
* Persistent Vector Database
* Query Rewriting
* Document Summarization
* Voice Input
* Docker Deployment

## Author

Akshay M
B.Tech Computer Science & Engineering
AI/ML Enthusiast

