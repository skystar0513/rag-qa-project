import pandas as pd

df = pd.read_csv("logs/test_log.csv", encoding="utf-8-sig")

# 전체 질문 수
print("전체 질문 수:", len(df))

# 고유 질문 수
print("고유 질문 수:", df["question"].nunique())