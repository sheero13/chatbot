import os

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Path to knowledge base folder
DATA_PATH = "../knowledge_base"

# Store all loaded documents
documents = []

# Load all .txt files
for file in os.listdir(DATA_PATH):

    if file.endswith(".txt"):

        file_path = os.path.join(DATA_PATH, file)

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        docs = loader.load()

        documents.extend(docs)

        print(f"Loaded: {file}")

print(f"\nTotal documents loaded: {len(documents)}")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

texts = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(texts)}")

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector database
db = Chroma.from_documents(
    texts,
    embeddings,
    persist_directory="vector_db"
)

print("\nVector DB created successfully!")