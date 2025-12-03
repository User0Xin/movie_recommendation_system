# -*- coding: utf-8 -*-
"""
基于社区检测的 MovieLens 推荐脚本（支持采样）

思路：
1. 复用 network_analysis.NetworkAnalyzer 构建用户-电影二部图；
2. 在「采样后的」二部图上对电影节点做加权投影，得到电影-电影图；
3. 在电影图上做社区检测（贪心模块度最大化）；
4. 对给定用户：
   - 看 TA 的高分电影主要落在哪些社区；
   - 选出“最偏好”的社区；
   - 在该社区内挑没有看过的电影做 Top-N 推荐。
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import community as nx_comm

# 保证可以导入 network_analysis.py
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from movie_recommendation_system.models.network_analysis import NetworkAnalyzer  # noqa: E402


class CommunityMovieLensAnalyzer(NetworkAnalyzer):
    """在 NetworkAnalyzer 基础上增加社区发现与推荐功能（支持采样）"""

    def __init__(self, ratings_path=None, movies_path=None):
        super().__init__(ratings_path=ratings_path, movies_path=movies_path)
        self.movie_projection = None
        self.movie_communities = None
        self.movie_to_community = None

        # 保存采样时实际使用的用户 / 电影节点（都是 "U_xxx" / "M_xxx"）
        self.sampled_user_nodes = None
        self.sampled_movie_nodes = None

    # ---------- 图 & 社区部分 ----------

    def build_movie_projection(
        self,
        min_shared_users: int = 3,
        user_sample_size: int | None = None,
        movie_sample_size: int | None = None,
        seed: int = 42,
    ):
        """
        构建「采样后的」电影投影图。

        Args:
            min_shared_users: 边的最小共同用户数（小于该值的边会被删掉）
            user_sample_size: 采样的用户节点数（None = 不采样，用全部用户）
            movie_sample_size: 采样的电影节点数（None = 不采样，用全部电影）
            seed: 随机种子
        """
        # 确保基础图已构建
        if self.G is None:
            self.load_data()
            self.build_bipartite_graph()

        rng = np.random.default_rng(seed)

        # 1) 选用户样本
        if user_sample_size is not None and user_sample_size < len(self.user_nodes):
            sampled_users = list(
                rng.choice(self.user_nodes, size=user_sample_size, replace=False)
            )
        else:
            sampled_users = list(self.user_nodes)

        # 2) 选电影样本
        if movie_sample_size is not None and movie_sample_size < len(self.movie_nodes):
            sampled_movies = list(
                rng.choice(self.movie_nodes, size=movie_sample_size, replace=False)
            )
        else:
            sampled_movies = list(self.movie_nodes)

        self.sampled_user_nodes = sampled_users
        self.sampled_movie_nodes = sampled_movies

        print("\n=== 构建采样后的电影投影图 ===")
        print(
            f"采样用户: {len(sampled_users)} / {len(self.user_nodes)}, "
            f"采样电影: {len(sampled_movies)} / {len(self.movie_nodes)}"
        )

        # 3) 在采样子图上做电影投影
        sub_nodes = sampled_users + sampled_movies
        subG = self.G.subgraph(sub_nodes)

        movie_graph = bipartite.weighted_projected_graph(subG, sampled_movies)
        print(
            f"投影后电影图: {movie_graph.number_of_nodes()} 节点, "
            f"{movie_graph.number_of_edges()} 边（未过滤）"
        )

        # 4) 根据共同用户数过滤弱边
        if min_shared_users > 1:
            to_remove = [
                (u, v)
                for u, v, d in movie_graph.edges(data=True)
                if d.get("weight", 0) < min_shared_users
            ]
            movie_graph.remove_edges_from(to_remove)
            print(
                f"按共同用户数 ≥ {min_shared_users} 过滤后: "
                f"{movie_graph.number_of_nodes()} 节点, "
                f"{movie_graph.number_of_edges()} 边"
            )

        self.movie_projection = movie_graph

    def detect_movie_communities(
        self,
        min_shared_users: int = 3,
        min_community_size: int = 10,
        user_sample_size: int | None = None,
        movie_sample_size: int | None = None,
        seed: int = 42,
    ):
        """
        在采样后的电影投影图上做社区检测。

        Args:
            min_shared_users: 电影投影图中边的最小共同用户数
            min_community_size: 过滤掉规模小于该值的社区
            user_sample_size: 采样用户数，例如 1000
            movie_sample_size: 采样电影数，例如 500
            seed: 随机种子
        """
        if self.movie_projection is None:
            self.build_movie_projection(
                min_shared_users=min_shared_users,
                user_sample_size=user_sample_size,
                movie_sample_size=movie_sample_size,
                seed=seed,
            )

        print("\n=== 社区检测（电影图，采样版） ===")
        Gm = self.movie_projection

        # 使用 NetworkX 自带的贪心模块度最大化算法
        raw_communities = list(
            nx_comm.greedy_modularity_communities(Gm, weight="weight")
        )

        # 过滤掉太小的社区
        communities = [c for c in raw_communities if len(c) >= min_community_size]

        self.movie_communities = communities
        movie_to_comm = {}
        for cid, comm in enumerate(communities):
            for node in comm:
                movie_to_comm[node] = cid
        self.movie_to_community = movie_to_comm

        print(
            f"检测到社区总数: {len(raw_communities)}，"
            f"过滤后（大小 ≥ {min_community_size}）: {len(communities)}"
        )

        return communities

    # ---------- 推荐 & 评分逻辑部分 ----------

    def _get_user_preferred_community(self, user_id, rating_threshold=4.0):
        """
        找到用户“最偏好”的电影社区：
        看看 TA 的高分电影主要集中在哪些社区。
        """
        if self.movie_to_community is None:
            # 如果还没做社区检测，用默认配置跑一遍
            self.detect_movie_communities()

        user_ratings = self.ratings[self.ratings["userId"] == user_id]

        # 高分电影
        liked = user_ratings[user_ratings["rating"] >= rating_threshold]["movieId"]

        liked_movie_nodes = [
            f"M_{mid}"
            for mid in liked
            if f"M_{mid}" in self.movie_to_community  # 只考虑采样图里存在的电影
        ]

        if not liked_movie_nodes:
            return None, Counter()

        counter = Counter()
        for node in liked_movie_nodes:
            cid = self.movie_to_community.get(node)
            if cid is not None:
                counter[cid] += 1

        if not counter:
            return None, Counter()

        best_cid = max(counter.items(), key=lambda x: x[1])[0]
        return best_cid, counter

    def _score_candidates(self, candidate_movie_ids):
        """
        给候选电影打分：结合平均评分 & 热度（评分次数的 log）。
        """
        df = self.ratings[self.ratings["movieId"].isin(candidate_movie_ids)]
        stats = (
            df.groupby("movieId")
            .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        stats["score"] = 0.7 * stats["avg_rating"] + 0.3 * np.log1p(
            stats["rating_count"]
        )
        return stats

    def score_movie_in_user_community(
        self,
        user_id: int,
        movie_id: int,
        rating_threshold: float = 4.0,
        min_shared_users: int = 3,
        min_community_size: int = 10,
        user_sample_size: int | None = 1000,
        movie_sample_size: int | None = 500,
        seed: int = 42,
    ) -> float:
        """
        给定用户ID和电影ID，返回一个“社区感知的评分”：

        优先级：
        1. 如果用户已经看过该电影并给出评分，直接返回用户真实评分（1~5）。
        2. 否则，如果该电影属于用户“最偏好”的电影社区：
              返回该电影的社区评分：
              score = 0.7 * avg_rating + 0.3 * log(1 + rating_count)
        3. 否则（不在偏好社区 / 不在社区图中等），返回 -1.0

        Args:
            user_id: 用户ID
            movie_id: 电影ID
            rating_threshold: 把用户评分 >= 该值的电影视为“喜欢”
            min_shared_users: 电影投影图边的最小共同用户数
            min_community_size: 最小社区大小
            user_sample_size: 社区检测阶段使用的用户样本数（默认 1000）
            movie_sample_size: 社区检测阶段使用的电影样本数（默认 500）
            seed: 随机种子

        Returns:
            float: 评分；若无法给出合理评分则返回 -1.0
        """
        # 确保数据、图已构建
        if not hasattr(self, "ratings"):
            self.load_data()
        if self.G is None:
            self.build_bipartite_graph()

        # 1️⃣ 如果用户已经评价过这部电影，优先返回用户给的评分
        user_movie_ratings = self.ratings[
            (self.ratings["userId"] == user_id)
            & (self.ratings["movieId"] == movie_id)
        ]
        if not user_movie_ratings.empty:
            # MovieLens 1M 理论上 user-movie 唯一，这里取第一条就可以
            user_rating = float(user_movie_ratings.iloc[0]["rating"])
            return user_rating

        # 2️⃣ 用户没看过：先做社区检测（采样版）
        self.detect_movie_communities(
            min_shared_users=min_shared_users,
            min_community_size=min_community_size,
            user_sample_size=user_sample_size,
            movie_sample_size=movie_sample_size,
            seed=seed,
        )

        # 找到用户最偏好的社区
        best_cid, counter = self._get_user_preferred_community(
            user_id, rating_threshold=rating_threshold
        )

        # 没有明显偏好社区，直接返回 -1
        if best_cid is None:
            return -1.0

        # 这个电影在不在社区映射里？
        movie_node = f"M_{movie_id}"
        movie_cid = self.movie_to_community.get(movie_node, None)
        if movie_cid is None:
            # 说明这个电影可能没被采样到，或者在被过滤掉的小社区里
            return -1.0

        # 如果电影不属于用户的“最偏好社区”，返回 -1
        if movie_cid != best_cid:
            return -1.0

        # 3️⃣ 属于用户偏好社区，计算它的“社区评分”
        df = self.ratings[self.ratings["movieId"] == movie_id]
        if df.empty:
            # 数据里都没有这个电影的评分，保底
            return -1.0

        stats = (
            df.groupby("movieId")
            .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        stats["score"] = 0.7 * stats["avg_rating"] + 0.3 * np.log1p(
            stats["rating_count"]
        )

        score = float(stats.iloc[0]["score"])
        return score

    def recommend_for_user(
        self,
        user_id: int,
        topn: int = 10,
        rating_threshold: float = 4.0,
        min_shared_users: int = 3,
        min_community_size: int = 10,
        user_sample_size: int | None = 1000,
        movie_sample_size: int | None = 500,
        seed: int = 42,
    ):
        """
        给指定用户做 Top-N 推荐（社区驱动 + 采样）。

        Args:
            user_id: 目标用户 ID
            topn: 推荐条数
            rating_threshold: 把用户评分 >= 该值的电影视为“喜欢”
            min_shared_users: 电影投影图边的最小共同用户数
            min_community_size: 最小社区大小
            user_sample_size: 社区检测阶段使用的用户样本数（默认 1000）
            movie_sample_size: 社区检测阶段使用的电影样本数（默认 500）
            seed: 随机种子
        """
        # 确保数据、图已构建
        if not hasattr(self, "ratings"):
            self.load_data()
        if self.G is None:
            self.build_bipartite_graph()

        # 先做一次社区检测（采样版）
        self.detect_movie_communities(
            min_shared_users=min_shared_users,
            min_community_size=min_community_size,
            user_sample_size=user_sample_size,
            movie_sample_size=movie_sample_size,
            seed=seed,
        )

        best_cid, counter = self._get_user_preferred_community(
            user_id, rating_threshold=rating_threshold
        )

        if best_cid is None:
            print("用户在采样社区中没有明显的高分电影，退化为全局热门推荐。")
            return self.popular_recommendation(user_id, topn=topn)

        print(f"\n用户 {user_id} 对社区的偏好统计: {dict(counter)}")
        print(f"选用社区 ID = {best_cid} 作为推荐依据。")

        # 该社区中的所有电影节点
        comm_nodes = self.movie_communities[best_cid]
        comm_movie_ids = [int(n.replace("M_", "")) for n in comm_nodes]

        # 用户已经看过的电影
        user_seen = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )

        # 候选 = 社区内电影 - 用户已看
        candidates = [mid for mid in comm_movie_ids if mid not in user_seen]

        if not candidates:
            print("该社区没有新的电影可推荐，退化为全局热门推荐。")
            return self.popular_recommendation(user_id, topn=topn)

        stats = self._score_candidates(candidates)
        stats = stats.sort_values("score", ascending=False).head(topn)

        result = stats.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        return result.to_dict(orient="records")

    def popular_recommendation(self, user_id, topn=10):
        """
        兜底策略：简单的“全局热门但用户没看过”的推荐。
        """
        user_seen = set(
            self.ratings[self.ratings["userId"] == user_id]["movieId"].tolist()
        )
        df = self.ratings[~self.ratings["movieId"].isin(user_seen)]
        stats = self._score_candidates(df["movieId"].unique())
        stats = stats.sort_values("score", ascending=False).head(topn)

        result = stats.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId", how="left"
        )
        return result.to_dict(orient="records")


# ---------- 你要的顶层函数接口（直接 import 用） ----------

def predict(
    user_id: int,
    movie_id: int,
    rating_threshold: float = 4.0,
    min_shared_users: int = 3,
    min_community_size: int = 10,
    user_sample_size: int | None = None,
    movie_sample_size: int | None = None,
    seed: int = 42,
) -> float:
    """
    顶层便捷函数：
    直接根据用户ID和电影ID，返回“社区感知评分”。

    使用方式：
        from network_community import get_movie_score_for_user
        score = get_movie_score_for_user(4169, 2858)

    返回规则：
    1. 如果该用户已经评分 → 返回用户真实评分 (1~5)。
    2. 否则，如果电影属于用户最偏好社区 → 返回社区评分
       (0.7 * 平均评分 + 0.3 * log(1 + 评分次数))
    3. 否则 → 返回 -1.0
    """
    analyzer = CommunityMovieLensAnalyzer()
    return analyzer.score_movie_in_user_community(
        user_id=user_id,
        movie_id=movie_id,
        rating_threshold=rating_threshold,
        min_shared_users=min_shared_users,
        min_community_size=min_community_size,
        user_sample_size=user_sample_size,
        movie_sample_size=movie_sample_size,
        seed=seed,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="基于社区检测的 MovieLens 推荐（支持采样）"
    )
    parser.add_argument("--user", type=int, required=True, help="需要推荐的用户 ID")
    parser.add_argument("--topn", type=int, default=10, help="推荐条数")

    # 社区 & 采样参数
    parser.add_argument(
        "--min-shared-users",
        type=int,
        default=3,
        help="电影投影图中，至少多少共同用户才连边（默认3）",
    )
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=10,
        help="过滤掉规模小于该值的电影社区（默认10）",
    )
    parser.add_argument(
        "--user-sample",
        type=int,
        default=1000,
        help="用于社区检测的用户采样数（默认1000，None 表示不用采样）",
    )
    parser.add_argument(
        "--movie-sample",
        type=int,
        default=500,
        help="用于社区检测的电影采样数（默认500，None 表示不用采样）",
    )
    args = parser.parse_args()

    # None 的处理：如果传负数就当成 None
    user_sample = args.user_sample if args.user_sample > 0 else None
    movie_sample = args.movie_sample if args.movie_sample > 0 else None

    analyzer = CommunityMovieLensAnalyzer()

    recs = analyzer.recommend_for_user(
        user_id=args.user,
        topn=args.topn,
        min_shared_users=args.min_shared_users,
        min_community_size=args.min_community_size,
        user_sample_size=user_sample,
        movie_sample_size=movie_sample,
    )

    print("\n=== 社区驱动推荐结果 ===")
    for i, r in enumerate(recs, 1):
        print(
            f"{i}. {r.get('title', 'N/A')} "
            f"(movieId={r['movieId']}) | "
            f"avg_rating={r['avg_rating']:.2f}, "
            f"count={r['rating_count']}, "
            f"score={r['score']:.3f} | "
            f"genres={r.get('genres', '')}"
        )


if __name__ == "__main__":
    main()
