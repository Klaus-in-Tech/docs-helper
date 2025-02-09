from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
import os
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from langchain_pinecone import PineconeEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import time
import httpcore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain




load_dotenv()
index_name = os.environ.get("PINECONE_INDEX_NAME")


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = PineconeEmbeddings(model="multilingual-e5-large")
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    chat = ChatOllama(model="mistral:latest", temperature=0.1)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")



    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")



    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )


    # Retry mechanism
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result = qa.invoke(input={"input": query, "chat_history": chat_history})
            break
        except httpcore.RemoteProtocolError as e:
            if attempt < max_retries - 1:
                time.sleep(2)  # Wait before retrying
                continue
            else:
                raise e
            
    new_result = {
        "query": result["input"],
        "result": result["answer"], 
        "source_documents": result["context"],
    }
    return new_result


if __name__ == "__main__":
    query = str(input("Enter your query: "))
    res = run_llm(query=query, chat_history=[])
    print(res["result"])
