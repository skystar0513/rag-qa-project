from langchain_text_splitters import CharacterTextSplitter

def split_documents(documents, chunk_size = 500, chunk_overlap=50):
    text_splitter = CharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap
    )

    docs = text_splitter.split_documents(documents)
    return docs