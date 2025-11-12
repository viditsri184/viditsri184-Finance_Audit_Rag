# api.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
import uvicorn
from federated_rag import load_faiss, load_chroma, make_ensemble_retriever, make_retrieval_chain
from ingest import split_docs, TextLoader, ensure_dir  # small reimport convenience
from memory_and_tools import SOXMemory, sec_update_tool
import redis

app = FastAPI()
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
rclient = redis.from_url(REDIS_URL)

# initialize stores lazily
FAISS_DIR = os.getenv("FAISS_DIR", "data/faiss_store")
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma_store")
sox_memory = SOXMemory()

def get_chain():
    faiss = load_faiss(FAISS_DIR)
    chroma = load_chroma(CHROMA_DIR)
    ensemble = make_ensemble_retriever(faiss, chroma)
    chain = make_retrieval_chain(ensemble)
    return chain

class QueryModel(BaseModel):
    query: str

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    # save temp
    contents = await file.read()
    tmp_path = r"C:\Users\User\Desktop\Capstone_project_Finance_audit_rag\tests\sample_sox.txt"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    # reuse ingest pipeline from ingest.py but simplified
    from ingest import load_text_files, split_docs, create_faiss, create_chroma
    docs = load_text_files([tmp_path])
    chunks = split_docs(docs)
    create_faiss(chunks, persist_dir=FAISS_DIR)
    create_chroma(chunks, persist_dir=CHROMA_DIR)
    return {"status": "ok", "chunks": len(chunks)}

@app.post("/query")
async def query(q: QueryModel):
    query_text = q.query.strip()
    # check redis cache
    cache_key = f"qa:{query_text}"
    cached = rclient.get(cache_key)
    if cached:
        return {"source": "cache", "answer": cached.decode("utf-8")}
    chain = get_chain()
    # run chain and save memory
    result = chain({"query": query_text})
    answer_text = result["result"] if isinstance(result, dict) and "result" in result else result
    # store in redis for 1 hour
    rclient.setex(cache_key, 3600, answer_text)
    sox_memory.save_context({"input": query_text}, {"output": answer_text})
    return {"source": "chain", "answer": answer_text, "sources": [d.metadata for d in result.get("source_documents", [])]}

@app.post("/update_sec")
async def update_sec(text: str = Form(...)):
    res = sec_update_tool({"chroma_dir": CHROMA_DIR}, text)
    return res


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
