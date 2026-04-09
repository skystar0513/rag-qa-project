import csv
from datetime import datetime
from pathlib import Path


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "qa_log.csv"

LOG_DIR.mkdir(exist_ok=True)


def save_log(question, answer, docs, scores):
    """
    질문/답변/근거 문서를 CSV로 저장한다.
    """
    file_exists = LOG_FILE.exists()

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "doc_count": len(docs),
        "scores": " | ".join([str(score) for score in scores]),
        "contexts": " | ".join(
            [doc.page_content.replace("\n", " ")[:300] for doc in docs]
        ),
    }

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "question",
                "answer",
                "doc_count",
                "scores",
                "contexts",
            ],
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)