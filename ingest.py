# ingest.py
import os
from pathlib import Path
from typing import List
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.embeddings import OpenAIEmbeddings

EMB_MODEL = "text-embedding-3-small"  # env OpenAI embedding key used by LangChain wrapper

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_text_files(paths: List[str]) -> List:
    docs = []
    for p in paths:
        loader = TextLoader(p, encoding="utf-8")
        docs.extend(loader.load())
    return docs

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

def create_faiss(docs, persist_dir="data/faiss_store"):
    ensure_dir(persist_dir)
    embeddings = OpenAIEmbeddings()
    faiss_store = FAISS.from_documents(docs, embeddings)
    faiss_store.save_local(persist_dir)
    return faiss_store

def create_chroma(docs, persist_dir="data/chroma_store"):
    ensure_dir(persist_dir)
    embeddings = OpenAIEmbeddings()
    chroma_store = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
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

    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")
    create_faiss(chunks, persist_dir=args.faiss_dir)
    create_chroma(chunks, persist_dir=args.chroma_dir)
    print("Done.")
