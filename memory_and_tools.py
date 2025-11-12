# memory_and_tools.py
import json
from typing import Dict, Any
from langchain.memory import BaseMemory
from langchain.schema import messages_from_dict, messages_to_dict

class SOXMemory(BaseMemory):
    """
    Simple custom memory that extracts 'SOX' occurrences into a dict:
    { 'controls': {control_name: details}, 'last_update': ts }
    and stores to memory.json
    """
    def __init__(self, path="data/sox_memory.json"):
        self.path = path
        try:
            with open(self.path, "r") as f:
                self.mem = json.load(f)
        except FileNotFoundError:
            self.mem = {"controls": {}, "metadata": {}}

    @property
    def memory_variables(self):
        return ["sox_memory"]

    def load_memory_variables(self, inputs):
        return {"sox_memory": self.mem}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        # naive extraction: look for lines containing "SOX" or "control", adjust with regex/NLP later
        text = inputs.get("input", "") or inputs.get("question", "")
        # simple heuristic:
        if "SOX" in text or "control" in text.lower():
            # store snippet keyed by a simple incremental id / short hash
            key = f"item_{len(self.mem['controls'])+1}"
            self.mem["controls"][key] = {"snippet": text}
            self.mem["metadata"]["last_update"] = __import__("datetime").datetime.utcnow().isoformat()
            with open(self.path, "w") as f:
                json.dump(self.mem, f, indent=2)

# Tool for mock SEC updates
def sec_update_tool(store_paths: Dict[str,str], update_text: str):
    """
    Simulate public SEC snippet update by adding a document to Chroma store (public mock).
    store_paths: dict with 'chroma_dir'
    """
    from langchain.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma

    chroma_dir = store_paths["chroma_dir"]
    ensure = __import__('pathlib').Path(chroma_dir).mkdir(parents=True, exist_ok=True)

    # create a temporary doc object
    from langchain.schema import Document
    doc = Document(page_content=update_text, metadata={"source": "mock_SEC_update"})
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents([doc])

    embeddings = OpenAIEmbeddings()
    chroma = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    chroma.add_documents(docs)
    chroma.persist()
    return {"status": "ok", "added_chunks": len(docs)}
