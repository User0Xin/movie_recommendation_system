# -*- coding: utf-8 -*-
"""
基于社区检测的 MovieLens 快速评分/推荐（离线预计算 + 在线快速查表版）

整体思路拆成两步：

1）离线阶段（一次性跑，比较慢，但只跑一次）：
    - 读取 ratings.dat / movies.dat；
    - 构建用户-电影二部图；
    - 在采样子图上做电影-电影投影 + 模块度贪心社区检测；
    - 得到：
        * 每部电影所属社区 movie_community.csv （movieId -> cid）
        * 每部电影的统计 + 社区评分 movie_stats.csv
          (movieId, rating_count, avg_rating, comm_score)
        * 每个用户的“最偏好社区” user_best_community.csv
          (userId -> best_cid)
    - 这些结果都丢到 output/community_precompute/ 下面。

2）在线阶段（大量 (user, movie) 打分时用，超快）：
    - 从上述三个 csv + 原始 ratings.dat 加载到内存；
    - 预先建好若干个 dict：
        * rating_lookup[(userId, movieId)] -> 用户真实评分
        * movie_cid[movieId]              -> 该电影所属社区 cid
        * user_best_cid[userId]           -> 该用户最偏好社区 best_cid
        * movie_score[movieId]            -> 电影的社区评分 comm_score
    - 对单个 (user, movie) 的预测逻辑：
        1. 如果用户已经看过这部电影 → 直接返回真实评分
        2. 否则查看该用户最偏好的社区 best_cid
        3. 看电影所在社区 cid 是否等于 best_cid
           - 相等 → 返回该电影的社区评分 comm_score
           - 不等 / 信息缺失 → 返回 -1（认为不适合推荐）

这样就可以支持“全量 user×item 遍历打分”，而不需要每次都重跑 NetworkX 的社区检测。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import community as nx_comm


# ----------------------------------------------------------------------
# 路径和配置：复用你之前的 FILE_PATH（指向 MovieLens 数据目录）
# ----------------------------------------------------------------------
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FILE_PATH  # 你原来就有的配置：FILE_PATH / "ratings.dat"


# ==============================
# 一、离线预计算部分
# ==============================

class OfflineCommunityPrecomputer:
    """
    离线预计算器：
    一次性从原始数据计算出：
      - 每部电影所属社区（movie_community.csv）
      - 每部电影的统计 + 社区评分（movie_stats.csv）
      - 每个用户的最偏好社区（user_best_community.csv）
    """

    def __init__(
        self,
        ratings_path: Path | None = None,
        movies_path: Path | None = None,
        output_dir: Path | None = None,
    ):
        if ratings_path is None:
            ratings_path = FILE_PATH / "ratings.dat"
        if movies_path is None:
            movies_path = FILE_PATH / "movies.dat"
        if output_dir is None:
            output_dir = PROJECT_ROOT / "output" / "community_precompute"

        self.ratings_path = Path(ratings_path)
        self.movies_path = Path(movies_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ratings: pd.DataFrame | None = None
        self.movies: pd.DataFrame | None = None
        self.G: nx.Graph | None = None
        self.user_nodes: list[str] | None = None
        self.movie_nodes: list[str] | None = None

    # ---------- 加载数据 & 构图 ----------

    def load_data(self):
        """加载 MovieLens 1M 的评分表和电影表"""
        print("正在加载数据...")
        self.ratings = pd.read_csv(
            self.ratings_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        self.movies = pd.read_csv(
            self.movies_path,
            sep="::",
            engine="python",
            names=["movieId", "title", "genres"],
            encoding="latin-1",
        )
        print(
            f"加载完成: {len(self.ratings)} 条评分记录, {len(self.movies)} 部电影"
        )

    def build_bipartite_graph(self):
        """构建用户-电影二部图 G"""
        print("正在构建用户-电影二部图...")
        G = nx.Graph()

        for _, row in self.ratings.iterrows():
            u = f"U_{row['userId']}"
            m = f"M_{row['movieId']}"
            G.add_node(u, node_type="user", id=row["userId"])
            G.add_node(m, node_type="movie", id=row["movieId"])
            # 边权重用评分（其实在投影时只关心“有没有边”，权重可以不用）
            G.add_edge(u, m, weight=row["rating"])

        self.G = G
        self.user_nodes = [n for n in G.nodes if n.startswith("U_")]
        self.movie_nodes = [n for n in G.nodes if n.startswith("M_")]

        print(
            f"图构建完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边"
        )
        print(
            f"用户节点: {len(self.user_nodes)}, 电影节点: {len(self.movie_nodes)}"
        )

    # ---------- 电影投影 & 社区检测 ----------

    def build_movie_projection(
        self,
        min_shared_users: int = 3,
        user_sample_size: int | None = 1000,
        movie_sample_size: int | None = 500,
        seed: int = 42,
    ) -> tuple[nx.Graph, list[str], list[str]]:
        """
        在采样后的二部图上构建电影-电影投影图（加权图，权重=共同用户数）。

        返回：
            movie_graph: 过滤弱边后的电影图
            sampled_users: 实际参与的用户节点 ID 列表（"U_xxx"）
            sampled_movies: 实际参与的电影节点 ID 列表（"M_xxx"）
        """
        assert self.G is not None, "需要先 build_bipartite_graph()"

        rng = np.random.default_rng(seed)

        # 采样用户
        if user_sample_size is not None and user_sample_size < len(self.user_nodes):
            sampled_users = list(
                rng.choice(self.user_nodes, size=user_sample_size, replace=False)
            )
        else:
            sampled_users = list(self.user_nodes)

        # 采样电影
        if movie_sample_size is not None and movie_sample_size < len(self.movie_nodes):
            sampled_movies = list(
                rng.choice(self.movie_nodes, size=movie_sample_size, replace=False)
            )
        else:
            sampled_movies = list(self.movie_nodes)

        print("\n=== 构建采样后的电影投影图 ===")
        print(
            f"采样用户: {len(sampled_users)} / {len(self.user_nodes)}, "
            f"采样电影: {len(sampled_movies)} / {len(self.movie_nodes)}"
        )

        # 在采样子图上做电影投影
        sub_nodes = sampled_users + sampled_movies
        subG = self.G.subgraph(sub_nodes)

        movie_graph = bipartite.weighted_projected_graph(subG, sampled_movies)
        print(
            f"投影后电影图: {movie_graph.number_of_nodes()} 节点, "
            f"{movie_graph.number_of_edges()} 边（未过滤）"
        )

        # 按共同用户数过滤弱边
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

        return movie_graph, sampled_users, sampled_movies

    def detect_movie_communities(
        self,
        movie_graph: nx.Graph,
        min_community_size: int = 10,
    ) -> dict[int, int]:
        """
        在电影-电影图上做社区检测（模块度贪心）。

        返回：
            movie_to_cid: dict[movieId -> cid]
        """
        print("\n=== 社区检测（电影图） ===")
        raw_communities = list(
            nx_comm.greedy_modularity_communities(movie_graph, weight="weight")
        )
        print(f"检测到社区总数: {len(raw_communities)}")

        # 过滤掉太小的社区
        big_comms = [c for c in raw_communities if len(c) >= min_community_size]
        print(
            f"过滤后（大小 ≥ {min_community_size}）: {len(big_comms)} 个社区"
        )

        movie_to_cid: dict[int, int] = {}
        for cid, comm in enumerate(big_comms):
            for node in comm:
                # node 形如 "M_2858"
                mid = int(node.replace("M_", ""))
                movie_to_cid[mid] = cid

        return movie_to_cid

    # ---------- 预计算三个表并保存 ----------

    def run_precompute(
        self,
        rating_threshold: float = 4.0,
        min_shared_users: int = 3,
        min_community_size: int = 10,
        user_sample_size: int | None = 1000,
        movie_sample_size: int | None = 500,
        seed: int = 42,
    ):
        """
        对整个数据集跑一遍预计算，并把结果存到 output_dir 下：

            movie_community.csv
            movie_stats.csv
            user_best_community.csv
        """
        print("=" * 60)
        print("开始离线预计算（社区 + 统计 + 用户偏好）")
        print("=" * 60)

        # 1. 加载数据 & 构二部图
        self.load_data()
        self.build_bipartite_graph()

        # 2. 构建电影投影图 + 社区检测
        movie_graph, sampled_users, sampled_movies = self.build_movie_projection(
            min_shared_users=min_shared_users,
            user_sample_size=user_sample_size,
            movie_sample_size=movie_sample_size,
            seed=seed,
        )
        movie_to_cid = self.detect_movie_communities(
            movie_graph, min_community_size=min_community_size
        )

        # 2.1 保存 movie_community.csv
        movie_community_df = pd.DataFrame(
            [{"movieId": mid, "cid": cid} for mid, cid in movie_to_cid.items()]
        )
        movie_community_path = self.output_dir / "movie_community.csv"
        movie_community_df.to_csv(movie_community_path, index=False)
        print(f"电影社区映射已保存到: {movie_community_path} "
              f"(共 {len(movie_community_df)} 部电影有社区标签)")

        # 3. 预计算每部电影的统计 + 社区评分
        print("\n=== 计算每部电影的统计信息与社区评分 ===")
        df = self.ratings
        movie_stats = (
            df.groupby("movieId")
            .agg(
                rating_count=("rating", "count"),
                avg_rating=("rating", "mean"),
            )
            .reset_index()
        )
        movie_stats["comm_score"] = 0.7 * movie_stats["avg_rating"] + 0.3 * np.log1p(
            movie_stats["rating_count"]
        )
        movie_stats_path = self.output_dir / "movie_stats.csv"
        movie_stats.to_csv(movie_stats_path, index=False)
        print(f"电影统计与社区评分已保存到: {movie_stats_path}")

        # 4. 预计算每个用户的最偏好社区 best_cid
        print("\n=== 计算每个用户的最偏好社区（best_cid） ===")
        # 把电影社区信息 merge 到评分表
        r = df.merge(movie_community_df, on="movieId", how="inner")
        # 只保留高分记录
        r_like = r[r["rating"] >= rating_threshold]

        if r_like.empty:
            print("⚠ 没有任何评分 ≥ 阈值，无法计算用户偏好社区")
            user_best_cid_df = pd.DataFrame(columns=["userId", "best_cid"])
        else:
            user_cid_cnt = (
                r_like.groupby(["userId", "cid"])
                .size()
                .reset_index(name="cnt")
            )
            # 对每个 user 取 cnt 最大的 cid
            idx = user_cid_cnt.groupby("userId")["cnt"].idxmax()
            user_best_cid_df = user_cid_cnt.loc[idx, ["userId", "cid"]].rename(
                columns={"cid": "best_cid"}
            )

        user_best_community_path = self.output_dir / "user_best_community.csv"
        user_best_cid_df.to_csv(user_best_community_path, index=False)
        print(
            f"用户最偏好社区已保存到: {user_best_community_path} "
            f"(共 {len(user_best_cid_df)} 个用户有偏好社区)"
        )

        print("\n预计算完成！")


# ==============================
# 二、在线快速评分 / 推荐部分
# ==============================

class CommunityFastScorer:
    """
    在线评分器：只依赖离线预计算好的三个 csv + 原始 ratings.dat，
    初始化时一次性加载进内存，之后对 (user, movie) 的预测就是几个字典查表。
    """

    def __init__(
        self,
        precompute_dir: Path | None = None,
        ratings_path: Path | None = None,
    ):
        if precompute_dir is None:
            precompute_dir = PROJECT_ROOT / "output" / "community_precompute"
        if ratings_path is None:
            ratings_path = FILE_PATH / "ratings.dat"

        self.precompute_dir = Path(precompute_dir)
        self.ratings_path = Path(ratings_path)

        # 数据表
        self.ratings: pd.DataFrame | None = None
        self.movie_stats: pd.DataFrame | None = None
        self.movie_community: pd.DataFrame | None = None
        self.user_best_community: pd.DataFrame | None = None

        # 快速查表的 dict
        self.rating_lookup: dict[tuple[int, int], float] = {}
        self.movie_cid: dict[int, int] = {}
        self.user_best_cid: dict[int, int] = {}
        self.movie_score: dict[int, float] = {}

        self._load_all()

    def _load_all(self):
        """一次性加载所有需要的数据，并构建查表字典"""
        print("\n=== 加载在线评分所需数据 ===")

        # 1. 加载原始评分表（只做一次）
        self.ratings = pd.read_csv(
            self.ratings_path,
            sep="::",
            engine="python",
            names=["userId", "movieId", "rating", "timestamp"],
            encoding="latin-1",
        )
        print(f"- 加载 ratings.dat: {len(self.ratings)} 条记录")

        # 2. 加载预计算结果
        movie_stats_path = self.precompute_dir / "movie_stats.csv"
        movie_community_path = self.precompute_dir / "movie_community.csv"
        user_best_community_path = self.precompute_dir / "user_best_community.csv"

        self.movie_stats = pd.read_csv(movie_stats_path)
        self.movie_community = pd.read_csv(movie_community_path)
        self.user_best_community = pd.read_csv(user_best_community_path)

        print(
            f"- 加载 movie_stats: {len(self.movie_stats)} 部电影统计信息\n"
            f"- 加载 movie_community: {len(self.movie_community)} 部电影有社区标签\n"
            f"- 加载 user_best_community: {len(self.user_best_community)} 个用户有偏好社区"
        )

        # 3. 构建查表字典
        print("- 构建查表字典 ...")

        # 3.1 用户真实评分 (userId, movieId) -> rating
        self.rating_lookup = {
            (int(row.userId), int(row.movieId)): float(row.rating)
            for row in self.ratings.itertuples(index=False)
        }

        # 3.2 电影 -> 社区 cid
        self.movie_cid = dict(
            zip(self.movie_community["movieId"].astype(int),
                self.movie_community["cid"].astype(int))
        )

        # 3.3 用户 -> 最偏好社区 best_cid
        self.user_best_cid = dict(
            zip(self.user_best_community["userId"].astype(int),
                self.user_best_community["best_cid"].astype(int))
        )

        # 3.4 电影 -> 社区评分 comm_score
        self.movie_score = dict(
            zip(self.movie_stats["movieId"].astype(int),
                self.movie_stats["comm_score"].astype(float))
        )

        print("  查表字典构建完成！")

    # ---------- 单点预测 ----------

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        对单个 (user, movie) 返回一个评分：

        1. 如果用户已经给过评分 → 返回真实评分（1~5）
        2. 否则，查用户最偏好社区 best_cid 和电影所属社区 cid：
            - 如果 cid == best_cid → 返回该电影的社区评分 comm_score
            - 否则 / 信息缺失 → -1.0
        """
        # 1. 用户是否已经看过 / 评分过
        key = (int(user_id), int(movie_id))
        if key in self.rating_lookup:
            return self.rating_lookup[key]

        # 2. 用户的最偏好社区
        best_cid = self.user_best_cid.get(int(user_id), None)
        if best_cid is None:
            return -1.0

        # 3. 电影所属社区
        cid = self.movie_cid.get(int(movie_id), None)
        if cid is None:
            return -1.0

        if cid != best_cid:
            return -1.0

        # 4. 返回该电影的社区评分
        score = self.movie_score.get(int(movie_id), None)
        if score is None:
            return -1.0
        return float(score)


# ==============================
# 三、模块级全局接口（方便直接 import 使用）
# ==============================

# 全局缓存一个 scorer，避免反复加载文件
_GLOBAL_SCORER_CACHE: dict[str, CommunityFastScorer] = {}


def get_global_scorer(precompute_dir: str | Path | None = None) -> CommunityFastScorer:
    """
    获取（或创建）一个全局 CommunityFastScorer 实例。
    多次调用会复用同一个对象，避免重复读盘。
    """
    if precompute_dir is None:
        precompute_dir = PROJECT_ROOT / "output" / "community_precompute"
    key = str(Path(precompute_dir).resolve())

    if key not in _GLOBAL_SCORER_CACHE:
        _GLOBAL_SCORER_CACHE[key] = CommunityFastScorer(precompute_dir=precompute_dir)
    return _GLOBAL_SCORER_CACHE[key]


def predict(
    user_id: int,
    movie_id: int,
    precompute_dir: str | Path | None = None,
) -> float:
    """
    你可以在其他地方直接：

        from network_community_fast import predict
        y = predict(4169, 2858)

    前提是：预计算已经跑过一次，在 precompute_dir 目录里有那三个 csv。
    """
    scorer = get_global_scorer(precompute_dir=precompute_dir)
    return scorer.predict(user_id, movie_id)


# ==============================
# 四、命令行入口（可选）
# ==============================

def main():
    """
    命令行用法示例：

    1）先做一次预计算（可以指定采样参数）：
        python network_community_fast.py --mode precompute \
            --user-sample 1000 --movie-sample 500 \
            --min-shared-users 3 --min-community-size 10

       如果想用全量用户/电影，就传 0 或负数：
        python network_community_fast.py --mode precompute \
            --user-sample 0 --movie-sample 0

    2）做一个测试预测：
        python network_community_fast.py --mode predict \
            --user 4169 --movie 2858
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="基于社区检测的 MovieLens 快速评分/推荐（离线预计算 + 在线查表）"
    )
    parser.add_argument(
        "--mode",
        choices=["precompute", "predict"],
        default="precompute",
        help="运行模式：precompute=离线预计算，predict=测试预测一个 (user, movie)",
    )

    # 预计算相关参数
    parser.add_argument("--rating-threshold", type=float, default=4.0,
                        help="判定“高分电影”的阈值（默认 4.0）")
    parser.add_argument("--min-shared-users", type=int, default=3,
                        help="电影投影图中共同用户数的下限（默认 3）")
    parser.add_argument("--min-community-size", type=int, default=10,
                        help="保留的最小社区规模（默认 10）")
    parser.add_argument("--user-sample", type=int, default=1000,
                        help="社区检测阶段用户采样数；<=0 表示全量")
    parser.add_argument("--movie-sample", type=int, default=500,
                        help="社区检测阶段电影采样数；<=0 表示全量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    # 预测模式参数
    parser.add_argument("--user", type=int, help="预测模式下的用户 ID")
    parser.add_argument("--movie", type=int, help="预测模式下的电影 ID")

    args = parser.parse_args()

    if args.mode == "precompute":
        # 采样参数统一处理：<=0 当成 None（表示全量）
        user_sample = args.user_sample if args.user_sample > 0 else None
        movie_sample = args.movie_sample if args.movie_sample > 0 else None

        pre = OfflineCommunityPrecomputer()
        pre.run_precompute(
            rating_threshold=args.rating_threshold,
            min_shared_users=args.min_shared_users,
            min_community_size=args.min_community_size,
            user_sample_size=user_sample,
            movie_sample_size=movie_sample,
            seed=args.seed,
        )
    else:
        if args.user is None or args.movie is None:
            raise ValueError("predict 模式需要提供 --user 和 --movie")
        y = predict(args.user, args.movie)
        print(f"用户 {args.user} 对电影 {args.movie} 的预测评分（社区模型）= {y}")


if __name__ == "__main__":
    main()
