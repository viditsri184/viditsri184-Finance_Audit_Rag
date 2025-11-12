import os
import json
from typing import Dict, Any
from datetime import datetime
from pathlib import Path

# Dummy base class so we don't depend on langchain.memory
class BaseMemory:
    """Lightweight BaseMemory stub for compatibility."""
    def load_memory_variables(self, inputs: Dict[str, Any]):
        raise NotImplementedError

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        raise NotImplementedError


class SOXMemory(BaseMemory):
    """
    Custom SOX Memory that extracts 'SOX' or 'control' mentions
    and persists them to a JSON file.
    """
    def __init__(self, path="data/sox_memory.json"):
        self.path = Path(path)
        if self.path.exists():
            with open(self.path, "r") as f:
                self.mem = json.load(f)
        else:
            self.mem = {"controls": {}, "metadata": {}}

    @property
    def memory_variables(self):
        return ["sox_memory"]

    def load_memory_variables(self, inputs):
        return {"sox_memory": self.mem}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        text = inputs.get("input", "") or inputs.get("question", "")
        if "SOX" in text or "control" in text.lower():
            key = f"item_{len(self.mem['controls']) + 1}"
            self.mem["controls"][key] = {"snippet": text}
            self.mem["metadata"]["last_update"] = datetime.utcnow().isoformat()

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "w") as f:
                json.dump(self.mem, f, indent=2)


def sec_update_tool(store_paths: Dict[str, str], update_text: str):
    """
    Simulate adding a new SEC update snippet into Chroma store.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import AzureOpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.documents import Document

    chroma_dir = store_paths["chroma_dir"]
    Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents([Document(page_content=update_text, metadata={"source": "mock_SEC_update"})])

    # Use Azure embedding model
    embeddings = AzureOpenAIEmbeddings(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    chroma = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    chroma.add_documents(docs)
    chroma.persist()

    return {"status": "ok", "added_chunks": len(docs)}
