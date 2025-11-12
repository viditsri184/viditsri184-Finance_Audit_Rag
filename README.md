🧾 Finance Audit RAG — Project

Federated Retrieval-Augmented Generation (RAG) for Financial Audits (SOX Filings)
Combines private and public knowledge bases with memory, update tools, caching, and a web dashboard.

📘 Overview

This project builds a federated RAG system that assists auditors in analyzing SOX filings and related compliance documents.

Core Features

Local FAISS store for private audit documents

Public Chroma store for mock SEC data

Federated retrieval via EnsembleRetriever (weighted FAISS + Chroma)

Custom memory tracking “SOX” control references

Bound tool for live mock SEC updates

FastAPI backend exposing /ingest, /query, /update_sec

Streamlit dashboard to view query history and responses

Extras:

Redis caching for query results

Pinecone integration for cloud-based vector retrieval

👥 Team & Responsibilities
Member	Task Area	File(s)	Key Components
Abhinav	Ingestion pipeline	ingest.py	TextLoader → Splitter → FAISS/Chroma persist
Vidit	Federated RAG logic	federated_rag.py	EnsembleRetriever + LCEL chain
Aditya	Memory + Tools	memory_and_tools.py	SOXMemory + SEC update tool
Virendra	Backend + UI + Extras	api.py, streamlit_dashboard.py	FastAPI, Streamlit, Redis, Pinecone
🏗️ Architecture
         ┌─────────────┐
         │   FastAPI    │
         │  (/query)    │
         └─────┬────────┘
               │
     ┌─────────▼──────────┐
     │  Federated RAG     │
     │  (EnsembleRetriever│
     │   FAISS + Chroma)  │
     └─────────┬──────────┘
               │
  ┌────────────┼───────────────┐
  │            │               │
FAISS (Private)   Chroma (Public SEC)
   │                     │
   └──► Federated Answer ◄──┘
          ▲
          │
   Memory (SOX dict) + SEC Update Tool
          │
    Cached via Redis
          │
   Displayed in Streamlit UI

⚙️ Setup Instructions
1️⃣ Create and activate a virtual environment

Windows (PowerShell):

python -m venv .venv
.venv\Scripts\activate


macOS / Linux:

python3 -m venv .venv
source .venv/bin/activate

2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

3️⃣ Set environment variables

Create a .env file in the project root:

OPENAI_API_KEY=sk-your-key-here
REDIS_URL=redis://localhost:6379/0
PINECONE_API_KEY=your-pinecone-key
PINECONE_ENV=us-east1-gcp

4️⃣ Ingest sample data

Put a sample file under tests/sample_sox.txt, then run:

python ingest.py --files tests/sample_sox.txt

5️⃣ Run the backend (FastAPI)
python api.py


Server starts on: http://localhost:8000

6️⃣ Test the endpoints

Upload & Ingest:

curl -X POST -F "file=@tests/sample_sox.txt" http://localhost:8000/ingest


Query (first time):

curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What are SOX controls for revenue recognition?"}'


Query again (cached via Redis):

curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"query":"What are SOX controls for revenue recognition?"}'


Mock SEC Update:

curl -X POST -F "text=New SEC guidance on audit trails" http://localhost:8000/update_sec

7️⃣ Launch the Streamlit dashboard
streamlit run streamlit_dashboard.py


Ask questions via the dashboard input.

View all query history with timestamps using st.dataframe.

🌩️ Optional: Pinecone & Redis Setup
Pinecone (Cloud Vector Store)

Create a free index at https://app.pinecone.io
.

Add keys to .env.

Use:

from langchain_pinecone import PineconeVectorStore
store = PineconeVectorStore.from_existing_index("audit-index", embedding=OpenAIEmbeddings())

Redis (Query Cache)

Run locally:

docker run -d --name redis -p 6379:6379 redis

🧠 Memory Behavior

SOXMemory automatically extracts mentions of “SOX” or “control” in queries and stores them in data/sox_memory.json for context-aware follow-ups.

Example entry:

{
  "controls": {
    "item_1": {"snippet": "What are SOX controls for revenue?"},
    "item_2": {"snippet": "List IT general controls under SOX."}
  },
  "metadata": {
    "last_update": "2025-11-12T10:33:00Z"
  }
}

🧪 Testing & Debug
Test	Command	Expected
Ingest	python ingest.py --files tests/sample_sox.txt	Creates data/faiss_store, data/chroma_store
Query	curl /query	Returns source=chain
Query (cached)	Repeat	Returns source=cache
Update SEC	curl /update_sec	Adds doc to Chroma
Streamlit	streamlit run streamlit_dashboard.py	Dashboard loads, history displayed
🧮 Evaluation Mapping (100 Marks)
Criterion	Description	Marks
Functionality	Federated RAG retrieval works (FAISS + Chroma)	40
Integration	Memory + Tool correctly integrated	20
Extras	Redis caching + Pinecone federation	20
Code/UI	Clean, modular code + Streamlit dashboard	10
Demo/Repo	Working demo + well-documented repo	10
Total		100
🧰 Project Structure
finance-audit-rag/
├─ ingest.py
├─ federated_rag.py
├─ memory_and_tools.py
├─ api.py
├─ streamlit_dashboard.py
├─ requirements.txt
├─ .env
├─ data/
│  ├─ faiss_store/
│  └─ chroma_store/
└─ tests/
   ├─ sample_sox.txt
   └─ run_tests.sh

🧑‍💻 Quick Demo Script
# Run full flow
python ingest.py --files tests/sample_sox.txt
python api.py &
streamlit run streamlit_dashboard.py


Then open: http://localhost:8501

🧱 Common Pitfalls
Issue	Fix
Chroma persist_directory not found	Ensure folder exists before writing
FAISS index not loading	Use same embeddings model when saving/loading
OpenAIAuthenticationError	Set OPENAI_API_KEY in .env
Redis connection refused	Start Redis container locally
Streamlit cannot reach API	Confirm API is running at http://localhost:8000
