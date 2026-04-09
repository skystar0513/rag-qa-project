import pandas as pd
from pathlib import Path


LOG_FILE = Path("logs/qa_log.csv")


def load_logs():
    """
    저장된 QA 로그 CSV를 읽어서 DataFrame으로 반환한다.
    파일이 없으면 빈 DataFrame 반환
    """
    if not LOG_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(LOG_FILE, encoding="utf-8-sig")
        return df
    except Exception:
        return pd.DataFrame()


def get_recent_logs(df, n=5):
    """
    최근 로그 n개 반환
    """
    if df.empty:
        return df

    return df.sort_values("timestamp", ascending=False).head(n)


def get_question_counts(df):
    """
    질문별 빈도수 집계
    """
    if df.empty or "question" not in df.columns:
        return pd.DataFrame()

    counts = (
        df["question"]
        .value_counts()
        .reset_index()
    )
    counts.columns = ["question", "count"]
    return counts


def get_basic_stats(df):
    """
    기본 통계 반환
    """
    if df.empty:
        return {
            "total_questions": 0,
            "unique_questions": 0,
        }

    total_questions = len(df)
    unique_questions = df["question"].nunique() if "question" in df.columns else 0

    return {
        "total_questions": total_questions,
        "unique_questions": unique_questions,
    }