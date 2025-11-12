# federated_rag.py
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.chains import RetrievalQA


# -----------------------------------------------------
# 🔹 Load environment variables
# -----------------------------------------------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT_LLM = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_GPT", "gpt-4o-mini")
AZURE_DEPLOYMENT_EMBED = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED", "text-embedding-3-large")


# -----------------------------------------------------
# 🔹 Load FAISS store
# -----------------------------------------------------
def load_faiss(persist_dir="data/faiss_store"):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_DEPLOYMENT_EMBED,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )
    if os.path.exists(persist_dir):
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    raise FileNotFoundError(f"❌ FAISS directory not found at {persist_dir}")


# -----------------------------------------------------
# 🔹 Load Chroma store
# -----------------------------------------------------
def load_chroma(persist_dir="data/chroma_store"):
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_DEPLOYMENT_EMBED,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
    )
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(f"❌ Chroma directory not found at {persist_dir}")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


# -----------------------------------------------------
# 🔹 Combine FAISS + Chroma into Ensemble Retriever
# -----------------------------------------------------
def make_ensemble_retriever(faiss_store, chroma_store, weights=(0.6, 0.4), k=6):
    retrievers = [
        faiss_store.as_retriever(search_kwargs={"k": k}),
        chroma_store.as_retriever(search_kwargs={"k": k}),
    ]
    return EnsembleRetriever(retrievers=retrievers, weights=list(weights))


# -----------------------------------------------------
# 🔹 Create RetrievalQA Chain with AzureChatOpenAI
# -----------------------------------------------------
def make_retrieval_chain(retriever):
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT_LLM,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.0,
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return qa


# -----------------------------------------------------
# 🔹 Main Execution
# -----------------------------------------------------
if __name__ == "__main__":
    print("🔹 Loading FAISS + Chroma retrievers using Azure OpenAI...")
    faiss = load_faiss()
    chroma = load_chroma()

    ensemble = make_ensemble_retriever(faiss, chroma)
    qa_chain = make_retrieval_chain(ensemble)

    query = "What are key SOX controls for revenue recognition?"
    print(f"\n🧠 Query: {query}")

    response = qa_chain.invoke({"query": query})
    print("\n💬 Answer:", response["result"])
    print("\n📚 Sources:", [d.metadata.get("source", "N/A") for d in response["source_documents"]])
