import os
import re

from langchain_core.documents import Document

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

# =========================================
# KNOWLEDGE BASE PATH
# =========================================

DATA_PATH = "../knowledge_base"

documents = []

# =========================================
# LOAD FILES
# =========================================

for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        file_path = os.path.join(DATA_PATH, file)

        with open(file_path, "r", encoding="utf-8") as f:

            text = f.read()

        # Split using Q:
        qa_pairs = re.split(r"\n(?=Q:)", text)

        for pair in qa_pairs:

            pair = pair.strip()

            if len(pair) > 20:

                documents.append(
                    Document(page_content=pair)
                )

        print(f"Loaded: {file}")

print(f"\nTotal Q&A documents: {len(documents)}")

# =========================================
# EMBEDDINGS
# =========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================================
# CREATE VECTOR DB
# =========================================

db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="vector_db"
)

print("\nVector DB created successfully!")