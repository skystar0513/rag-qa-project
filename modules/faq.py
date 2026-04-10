import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings


LOG_FILE = Path("logs/qa_log.csv")


def load_logs():
    """
    로그 CSV를 읽어 DataFrame으로 반환한다.
    파일이 없으면 빈 DataFrame 반환
    """
    if not LOG_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(LOG_FILE, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def group_similar_questions(questions, threshold=0.88):
    """
    질문 리스트를 임베딩 기반으로 유사 질문끼리 그룹화한다.
    """
    if not questions:
        return []

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectors = embeddings.embed_documents(questions)
    sim_matrix = cosine_similarity(vectors)

    visited = [False] * len(questions)
    groups = []

    for i in range(len(questions)):
        if visited[i]:
            continue

        current_group = [i]
        visited[i] = True

        for j in range(i + 1, len(questions)):
            if not visited[j] and sim_matrix[i][j] >= threshold:
                current_group.append(j)
                visited[j] = True

        groups.append(current_group)

    return groups


def get_semantic_faq_answers(df, threshold=0.88, top_n=5):
    """
    의미가 유사한 질문끼리 그룹화한 뒤,
    각 그룹의 대표 질문 / 반복 횟수 / 대표 답변 반환
    """
    if df.empty or "question" not in df.columns:
        return pd.DataFrame(columns=["question", "count", "answer", "grouped_questions"])

    questions = df["question"].dropna().tolist()
    if not questions:
        return pd.DataFrame(columns=["question", "count", "answer", "grouped_questions"])

    groups = group_similar_questions(questions, threshold=threshold)

    faq_rows = []

    for group in groups:
        grouped_questions = [questions[idx] for idx in group]

        # 대표 질문: 가장 짧은 질문을 대표로 사용
        representative_question = min(grouped_questions, key=len)

        # 해당 그룹에 속하는 질문들의 로그만 추출
        group_df = df[df["question"].isin(grouped_questions)].copy()

        # 가장 최근 답변을 대표 답변으로 사용
        if "timestamp" in group_df.columns:
            group_df = group_df.sort_values("timestamp", ascending=False)

        representative_answer = (
            group_df.iloc[0]["answer"]
            if not group_df.empty and "answer" in group_df.columns
            else "답변 없음"
        )

        faq_rows.append({
            "question": representative_question,
            "count": len(grouped_questions),
            "answer": representative_answer,
            "grouped_questions": grouped_questions
        })

    faq_df = pd.DataFrame(faq_rows)

    # 유사 질문 개수가 많은 순으로 정렬
    faq_df = faq_df.sort_values("count", ascending=False).head(top_n)

    return faq_df