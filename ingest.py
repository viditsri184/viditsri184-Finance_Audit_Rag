# ingest.py
import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv

# Load environment variables for Azure
load_dotenv()

# ✅ Updated imports for LangChain v1.0+
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Use your Azure embedding model
EMB_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED", "text-embedding-3-large")
print(EMB_MODEL)

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_text_files(paths: List[str]):
    """Load text files from provided paths."""
    docs = []
    for p in paths:
        loader = TextLoader(p, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def split_docs(docs):
    """Split long documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def create_faiss(docs, persist_dir="data/faiss_store"):
    """Create and persist a FAISS store using Azure embeddings."""
    ensure_dir(persist_dir)
    embeddings = AzureOpenAIEmbeddings(model=EMB_MODEL)
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_store.save_local(persist_dir)
    return faiss_store

def create_chroma(docs, persist_dir="data/chroma_store"):
    """Create and persist a Chroma store using Azure embeddings."""
    ensure_dir(persist_dir)
    embeddings = AzureOpenAIEmbeddings(model=EMB_MODEL)
    chroma_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    chroma_store.persist()
    return chroma_store

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--faiss_dir", default="data/faiss_store")
    parser.add_argument("--chroma_dir", default="data/chroma_store")
    args = parser.parse_args()

    docs = load_text_files(args.files)
    chunks = split_docs(docs)

    print(f"📄 Loaded {len(docs)} docs → 🔹 {len(chunks)} chunks")
    create_faiss(chunks, persist_dir=args.faiss_dir)
    create_chroma(chunks, persist_dir=args.chroma_dir)
    print("✅ Done creating FAISS + Chroma vector stores.")
