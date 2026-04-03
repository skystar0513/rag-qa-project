from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def build_vectorstore(docs):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore