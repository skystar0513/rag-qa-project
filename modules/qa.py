from langchain_openai import ChatOpenAI


def ask_question(query, retriever):
    relevant_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = f"""
다음 문서를 기반으로 질문에 답하라.
문서에 없는 내용은 추측하지 말고 모른다고 답하라.
답변은 간결하고 명확하게 작성하라.

문서:
{context}

질문:
{query}
"""

    response = llm.invoke(prompt)
    return response.content, relevant_docs