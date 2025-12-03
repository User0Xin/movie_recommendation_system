# simple_predict.py
import pandas as pd
from pathlib import Path

# 修改成你自己的 user_movie_scores.csv 路径
SCORES_PATH = Path("output/user_movie_scores.csv")


def predict(user_id: int, movie_id: int) -> float:
    """
    输入：
        user_id: 用户ID
        movie_id: 电影ID

    输出：
        预计算好的评分：
        - 如果 user_movie_scores.csv 里有这一行 → 返回 pred_score
        - 否则 → 返回 -1.0
    """
    # 每次调用都简单读一次表（最简单写法）
    df = pd.read_csv(SCORES_PATH)

    # 根据 userId + movieId 精确匹配
    row = df[(df["userId"] == user_id) & (df["movieId"] == movie_id)]

    if row.empty:
        return -1.0

    return float(row["score"].iloc[0])
