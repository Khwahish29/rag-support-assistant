"""
create_vector_store.py

Generates embeddings from cleaned support tickets and stores them in
ChromaDB persistently in 'vector_store/' folder.

Features:
- Persistent storage (won't lose embeddings after restart)
- Batch embedding for speed
- Duplicate check using metadata
"""

from sentence_transformers import SentenceTransformer
import chromadb
import pandas as pd
from tqdm import tqdm
import os

# -------------------------------
# 1️⃣ Load cleaned data
# -------------------------------
df = pd.read_csv("data/cleaned_tickets.csv")

if "text" not in df.columns:
    raise ValueError("Missing 'text' column in cleaned_tickets.csv. Run data_preparation.py first!")

# Reset index for unique IDs
df = df.reset_index(drop=True)

# -------------------------------
# 2️⃣ Initialize embedding model
# -------------------------------
print("🔍 Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# -------------------------------
# 3️⃣ Initialize persistent ChromaDB
# -------------------------------
os.makedirs("vector_store", exist_ok=True)
client = chromadb.PersistentClient(path="vector_store")
collection = client.get_or_create_collection("support_tickets")

# -------------------------------
# 4️⃣ Prepare existing IDs to avoid duplicates
# -------------------------------
existing_ids = set()
metadatas = collection.get(include=['metadatas'])['metadatas']
for meta in metadatas:
    existing_ids.add(meta['ticket_id'])

# -------------------------------
# 5️⃣ Generate embeddings in batches
# -------------------------------
print("🧠 Generating embeddings and storing in ChromaDB...")

batch_size = 32
texts = df["text"].tolist()

for start_idx in tqdm(range(0, len(texts), batch_size)):
    end_idx = min(start_idx + batch_size, len(texts))
    batch_texts = texts[start_idx:end_idx]
    embeddings = model.encode(batch_texts)

    for i, emb in enumerate(embeddings):
        idx = start_idx + i
        doc_id = str(df.loc[idx, "Ticket ID"]) if "Ticket ID" in df.columns else str(idx)

        if doc_id in existing_ids:
            continue  # skip already stored tickets

        collection.add(
            documents=[df.loc[idx, "text"]],
            embeddings=[emb],
            metadatas=[{"ticket_id": doc_id}],
            ids=[doc_id]
        )
        existing_ids.add(doc_id)  # update set to avoid duplicates in same run

print("✅ All embeddings stored in ChromaDB successfully!")
print(f"📊 Total vectors stored: {collection.count()}")
