import pandas as pd
from pathlib import Path

LOG_FILE = Path("logs/qa_log.csv")

import pandas as pd
from pathlib import Path


LOG_FILE = Path("logs/qa_log.csv")


def load_logs():
    
    #로그 CSV를 읽어 DataFrame으로 반환한다. 파일이 없으면 빈 DataFrame 반환
    
    if not LOG_FILE.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(LOG_FILE, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def get_top_questions(df, top_n=5):
    
    # 자주 나온 질문 Top N 반환
    
    if df.empty or "question" not in df.columns:
        return pd.DataFrame(columns=["question", "count"])

    faq_df = (
        df["question"]
        .value_counts()
        .reset_index()
    )
    faq_df.columns = ["question", "count"]

    return faq_df.head(top_n)


def get_faq_answers(df, top_n=5):
    
    # 자주 나온 질문 Top N에 대해 가장 최근 답변 1개를 매칭해서 반환

    if df.empty or "question" not in df.columns or "answer" not in df.columns:
        return pd.DataFrame(columns=["question", "count", "answer"])

    top_questions = get_top_questions(df, top_n=top_n)

    if top_questions.empty:
        return pd.DataFrame(columns=["question", "count", "answer"])

    if "timestamp" in df.columns:
        df_sorted = df.sort_values("timestamp", ascending=False)
    else:
        df_sorted = df.copy()

    latest_answers = (
        df_sorted[["question", "answer"]]
        .drop_duplicates(subset=["question"], keep="first")
    )

    result = top_questions.merge(latest_answers, on="question", how="left")
    return result