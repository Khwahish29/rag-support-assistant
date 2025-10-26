"""
data_preparation.py

Cleans and prepares the Customer Support Ticket dataset for RAG (Retrieval-Augmented Generation).

Dataset columns include:
Ticket ID, Customer Name, Customer Email, Ticket Description, Resolution, etc.

Steps:
1. Load dataset
2. Remove PII (names, emails)
3. Drop duplicates/missing rows
4. Clean and combine text
5. Save cleaned CSV for RAG embedding
"""

import pandas as pd
import re
from langdetect import detect

# -------------------------------
# 🧹 Text Cleaning Function
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    text = text.lower()

    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove phone numbers
    text = re.sub(r'\b\d{10,12}\b', '', text)

    # Remove names following greetings
    text = re.sub(r'\b(dear|hi|hello)\s+[a-z]+\b', '', text)

    # Remove ticket IDs, hashtags, brackets
    text = re.sub(r'(\[.*?\]|\#\d+)', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# -------------------------------
# ⚙️ Load Dataset
# -------------------------------
input_path = "data/customer_support_tickets.csv"
output_path = "data/cleaned_tickets.csv"

print("📂 Loading dataset...")
df = pd.read_csv(input_path)
print(f"✅ Loaded {len(df)} rows and {len(df.columns)} columns.")

# -------------------------------
# 🧾 Drop Irrelevant Columns (PII)
# -------------------------------
columns_to_drop = ["Customer Name", "Customer Email"]
df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

# -------------------------------
# 🧾 Handle Missing Values
# -------------------------------
if "Ticket Description" not in df.columns or "Resolution" not in df.columns:
    raise KeyError("Expected columns 'Ticket Description' and 'Resolution' not found. Check dataset headers!")

df = df.dropna(subset=["Ticket Description", "Resolution"])

# -------------------------------
# 🧽 Remove Duplicates
# -------------------------------
df = df.drop_duplicates(subset=["Ticket Description", "Resolution"])
print(f"🧹 After dropping duplicates: {len(df)} rows remain.")

# -------------------------------
# 🧠 Clean Text
# -------------------------------
print("🧠 Cleaning text fields...")
df["Ticket Description"] = df["Ticket Description"].astype(str).apply(clean_text)
df["Resolution"] = df["Resolution"].astype(str).apply(clean_text)

# Combine for embeddings
df["text"] = df["Ticket Description"] + " " + df["Resolution"]

# -------------------------------
# 🌍 Keep English-only entries
# -------------------------------
print("🌍 Filtering non-English entries...")
def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

df = df[df["text"].apply(is_english)]
print(f"✅ English-only rows: {len(df)}")

# -------------------------------
# ⚠️ Remove irrelevant/test entries
# -------------------------------
df = df[~df["text"].str.contains("test|dummy|n/a", case=False)]
print(f"🧽 After removing irrelevant entries: {len(df)} rows remain.")

# -------------------------------
# ✂️ Trim long text
# -------------------------------
df["text"] = df["text"].apply(lambda x: x[:1000])

# -------------------------------
# 💾 Save Cleaned Dataset
# -------------------------------
df.to_csv(output_path, index=False)
print(f"✅ Cleaned dataset saved at: {output_path}")
print(f"📊 Final dataset shape: {df.shape}")
print("🎉 Data preparation complete! Ready for embedding.")
