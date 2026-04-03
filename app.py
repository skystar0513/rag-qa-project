from dotenv import load_dotenv

from modules.loader import load_pdf
from modules.splitter import split_documents
from modules.vectorstore import build_vectorstore
from modules.qa import ask_question

def main():
    load_dotenv()

    file_path = 'data/sample.pdf'

    print("PDF 로딩 중....")
    documents = load_pdf(file_path)

    print("문서 분할 중...")
    docs = split_documents(documents)

    docs = docs[:50] #테스트를 위해 일부만 사용

    print("벡터 스토어 생성 중...")
    vectorstore = build_vectorstore(docs)

    retriever = vectorstore.as_retriever()

    while True:
        query = input("\n 질문을 입력하세요 (종료: exit):")
        if query.lower() == "exit":
            print("프로그램을 종료합니다.")
            break
        
        answer, relevant_docs = ask_question(query, retriever)

        print("\n[답변]")
        print(answer)

        print("\n[참고 문서]")
        for i, doc in enumerate(relevant_docs):
            print(f"\n--- 문서 {i+1} ---")
            print(doc.page_content[:300])

if __name__ == "__main__":
    main()