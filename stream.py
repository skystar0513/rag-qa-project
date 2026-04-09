import os
import tempfile
from dotenv import load_dotenv
import streamlit as st

from modules.loader import load_pdf
from modules.splitter import split_documents
from modules.vectorstore import build_vectorstore
from modules.qa import ask_question
from modules.logger import save_log
from modules.log_reader import load_logs, get_recent_logs, get_question_counts, get_basic_stats
from modules.faq import load_logs, get_faq_answers


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
                answer, relevant_docs, scores = ask_question(query, st.session_state.retriever)
                save_log(query, answer, relevant_docs, scores)

            st.subheader("답변")
            st.write(answer)

            st.success("질문/답변 로그가 저장되었습니다.")

            st.subheader("참고 문서")
            for i, (doc, score) in enumerate(zip(relevant_docs, scores), start=1):
                with st.expander(f"문서 {i}"):
                    st.write(f"**유사도 점수:** {score}")
                    st.write(doc.page_content)
                    if hasattr(doc, "metadata") and doc.metadata:
                        st.caption(f"metadata: {doc.metadata}")
        except Exception as e:
            st.error(f"답변 생성 중 오류가 발생했습니다: {e}")
        
        st.divider()
        st.subheader("질문 로그 분석")

        logs_df = load_logs()

        if logs_df.empty:
            st.info("아직 저장된 로그가 없습니다.")
        else:
            stats = get_basic_stats(logs_df)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("총 질문 수", stats["total_questions"])
            with col2:
                st.metric("고유 질문 수", stats["unique_questions"])

            st.markdown("### 최근 질문 로그")
            recent_logs = get_recent_logs(logs_df, n=5)

            for i, row in recent_logs.iterrows():
                with st.expander(f"[{row['timestamp']}] {row['question']}"):
                    st.write(f"**답변:** {row['answer']}")
                    if "scores" in row:
                        st.write(f"**scores:** {row['scores']}")
                    if "contexts" in row:
                        st.write(f"**contexts:** {row['contexts']}")

            st.markdown("### 자주 나온 질문")
            question_counts = get_question_counts(logs_df)

            if not question_counts.empty:
                st.dataframe(question_counts.head(10), use_container_width=True)

st.divider()
st.subheader("자동 FAQ")

faq_logs = load_logs()
faq_df = get_faq_answers(faq_logs, top_n=3)

if faq_df.empty:
    st.info("FAQ를 생성할 로그가 아직 부족합니다.")
else:
    st.caption("저장된 질문 로그를 기준으로 자주 나온 질문을 FAQ 형태로 정리했습니다.")

    for i, row in faq_df.iterrows():
        with st.expander(f"Q. {row['question']}  (반복 {row['count']}회)"):
            st.write(f"**A.** {row['answer']}")