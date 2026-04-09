from langchain_openai import ChatOpenAI


def ask_question(query, retriever, k=4):
    """
    사용자 질문에 대해 관련 문서를 검색하고,
    검색된 문서를 바탕으로 답변을 생성한다.

    반환값:
        answer (str): LLM이 생성한 답변
        relevant_docs (list): 검색된 문서 리스트
        scores (list): 각 문서의 유사도 점수
    """
    # retriever에서 직접 점수는 못 받으므로 vectorstore에서 가져옴
    results = retriever.vectorstore.similarity_search_with_score(query, k=k)

    relevant_docs = [doc for doc, score in results]
    scores = [score for doc, score in results]

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
    return response.content, relevant_docs, scores