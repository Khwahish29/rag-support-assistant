import os
import json
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def load_tickets():
    path = os.path.join(os.path.dirname(__file__), "../data/tickets.json")
    with open(path, "r") as f:
        tickets = json.load(f)
    docs = []
    for t in tickets:
        content = f"{t['title']} - {t['description']} Resolution: {t['resolution']}"
        docs.append(Document(page_content=content, metadata={"ticket_id": t["ticket_id"], "sql": t["sql"]}))
    return docs


def ingest():
    tickets = load_tickets()
    embeddings = OpenAIEmbeddings()  # requires OPENAI_API_KEY env var
    db = Chroma.from_documents(tickets, embeddings, persist_directory="../chroma_db")
    db.persist()
    print("✅ Ingestion complete!")

if __name__ == "__main__":
    ingest()
