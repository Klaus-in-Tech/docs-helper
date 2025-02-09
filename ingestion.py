import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader

load_dotenv()
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

index_name = os.environ.get("PINECONE_INDEX_NAME")


def ingest_docs():
    loader = ReadTheDocsLoader(path="langchain-docs", encoding="utf-8")

    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs\\", "https://")
        doc.metadata.update({"source": new_url})
        
    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(documents, embeddings, index_name=index_name)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
