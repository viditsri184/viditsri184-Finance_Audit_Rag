from openai import AzureOpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

response = client.embeddings.create(
    model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME_EMBED"),
    input="test embedding"
)

print(response.data[0].embedding[:5])  # should print some floats

