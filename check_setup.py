import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()
print("OPENAI_API_KEY set:", bool(os.getenv("OPENAI_API_KEY")))
client = QdrantClient(url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"])
print("Qdrant collections:", client.get_collections())