import pandas as pd
#from langchain_openai import OpenAIEmbeddings api사용X
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()


def group_similar_questions(questions, threshold=0.9):
    """
    질문 리스트를 임베딩 기반으로 유사 질문끼리 그룹화한다.
    threshold 이상이면 같은 그룹으로 묶는다.
    """
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small") api사용X
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectors = embeddings.embed_documents(questions)

    from sklearn.metrics.pairwise import cosine_similarity
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

    return groups, sim_matrix


def main():
    df = pd.read_csv("logs/test_log.csv", encoding="utf-8-sig")

    if "question" not in df.columns:
        raise ValueError("CSV에 question 컬럼이 없습니다.")

    questions = df["question"].dropna().tolist()

    print(f"전체 질문 수: {len(questions)}")

    groups, sim_matrix = group_similar_questions(questions, threshold=0.9)

    print(f"그룹화 후 질문 수: {len(groups)}")
    print(f"줄어든 질문 수: {len(questions) - len(groups)}")

    print("\n=== 그룹 결과 ===")
    for idx, group in enumerate(groups, start=1):
        print(f"\n[그룹 {idx}]")
        for q_idx in group:
            print(f"- {questions[q_idx]}")


if __name__ == "__main__":
    main()