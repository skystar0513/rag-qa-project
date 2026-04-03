import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from modules.loader import load_pdf
from modules.splitter import split_documents
from modules.vectorstore import build_vectorstore
from modules.qa import ask_question


load_dotenv()

st.set_page_config(page_title="RAG 문서 QA 시스템", page_icon="📄", layout="wide")

st.title("📄 RAG 기반 문서 QA 시스템")
st.caption("PDF 문서를 업로드하고 질문하면, 관련 내용을 검색한 뒤 문서 기반으로 답변합니다.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "file_name" not in st.session_state:
    st.session_state.file_name = None

with st.sidebar:
    st.header("설정")
    uploaded_file = st.file_uploader("PDF 업로드", type=["pdf"])
    chunk_size = st.slider("Chunk 크기", min_value=300, max_value=1500, value=500, step=100)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=300, value=50, step=10)
    doc_limit = st.slider("테스트용 최대 chunk 수", min_value=10, max_value=300, value=50, step=10)
    process_btn = st.button("문서 처리 시작", use_container_width=True)

    st.divider()
    st.markdown("### 안내")
    st.markdown(
        "- PDF를 업로드한 뒤 **문서 처리 시작**을 누르세요.\n"
        "- 문서 처리 후 질문을 입력하면 답변이 생성됩니다.\n"
        "- 하단에서 참고 문서 chunk도 함께 확인할 수 있습니다."
    )

if process_btn:
    if uploaded_file is None:
        st.warning("먼저 PDF 파일을 업로드해 주세요.")
    elif not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
    else:
        try:
            with st.spinner("PDF를 읽고 벡터스토어를 생성하는 중입니다..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                documents = load_pdf(temp_path)
                docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                docs = docs[:doc_limit]

                vectorstore = build_vectorstore(docs)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.processed = True
                st.session_state.file_name = uploaded_file.name

                os.remove(temp_path)

            st.success(f"문서 처리 완료: {uploaded_file.name}")
        except Exception as e:
            st.error(f"문서 처리 중 오류가 발생했습니다: {e}")

left, right = st.columns([1.2, 1])

with left:
    st.subheader("질문 입력")
    query = st.text_input("문서에 대해 궁금한 점을 입력하세요.", placeholder="예: 제품 사용 시 주의해야 할 안전사항은 무엇인가?")
    ask_btn = st.button("답변 생성", type="primary", use_container_width=True)

    if st.session_state.processed:
        st.info(f"현재 문서: {st.session_state.file_name}")
    else:
        st.info("왼쪽 사이드바에서 PDF를 업로드하고 처리해 주세요.")

with right:
    st.subheader("현재 상태")
    st.metric("문서 처리 상태", "완료" if st.session_state.processed else "대기")
    st.metric("질문 가능 여부", "가능" if st.session_state.retriever else "불가")

if ask_btn:
    if not st.session_state.retriever:
        st.warning("먼저 PDF를 업로드하고 문서를 처리해 주세요.")
    elif not query.strip():
        st.warning("질문을 입력해 주세요.")
    else:
        try:
            with st.spinner("답변을 생성하는 중입니다..."):
                answer, relevant_docs = ask_question(query, st.session_state.retriever)

            st.subheader("답변")
            st.write(answer)

            st.subheader("참고 문서")
            for i, doc in enumerate(relevant_docs, start=1):
                with st.expander(f"문서 {i}"):
                    st.write(doc.page_content)
                    if hasattr(doc, "metadata") and doc.metadata:
                        st.caption(f"metadata: {doc.metadata}")
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
