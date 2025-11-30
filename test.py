import pandas as pd
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
import community  # python-louvain

# 修改成你下载后的路径
# 读取dat格式数据
ratings = pd.read_csv("./datas/ml-1m/ml-1m/ratings.csv")
movies = pd.read_csv("./datas/ml-1m/ml-1m/movies.csv")


def build_bipartite_graph(ratings):
    G = nx.Graph()
    for _, row in ratings.iterrows():
        u = f"U_{row['userId']}"
        i = f"I_{row['movieId']}"
        G.add_edge(u, i, weight=row['rating'])
    return G


G = build_bipartite_graph(ratings)
print("Graph built. Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())


def build_transition_matrix(G):
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    rows, cols, data = [], [], []

    for u, v, d in G.edges(data=True):
        w = d['weight']
        rows.append(idx[u]);
        cols.append(idx[v]);
        data.append(w)
        rows.append(idx[v]);
        cols.append(idx[u]);
        data.append(w)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    col_sum = np.array(A.sum(axis=0)).reshape(-1)
    col_sum[col_sum == 0] = 1
    W = A / col_sum  # column normalization
    return W, nodes, idx


W, nodes, idx = build_transition_matrix(G)


def rwr(W, idx, nodes, source_node, alpha=0.15, max_iter=100, tol=1e-8):
    N = W.shape[0]
    p = np.zeros(N)
    e = np.zeros(N)
    e[idx[source_node]] = 1
    p = e.copy()

    for _ in range(max_iter):
        p_new = (1 - alpha) * (W @ p) + alpha * e
        if np.linalg.norm(p_new - p, 1) < tol:
            break
        p = p_new
    return p


def recommend_rwr(G, W, idx, nodes, user_id, K=10):
    source = f"U_{user_id}"
    score = rwr(W, idx, nodes, source)

    # 过滤掉用户已看过的
    seen_items = set(v for u, v in G.edges(source))
    result = []

    for node, s in zip(nodes, score):
        if node.startswith("I_") and node not in seen_items:
            result.append((node, s))

    result.sort(key=lambda x: x[1], reverse=True)
    return result[:K]


def detect_communities(G):
    partition = community.best_partition(G)  # 返回字典 node → community id
    return partition


partition = detect_communities(G)


def recommend_community(G, partition, user_id, K=10):
    u = f"U_{user_id}"
    user_comm = partition[u]

    # 找到该社区中的电影节点
    movies_in_comm = [n for n, c in partition.items() if c == user_comm and n.startswith("I_")]

    # 去除已看
    seen = set(v for u2, v in G.edges(u))
    candidates = [m for m in movies_in_comm if m not in seen]

    # 评分权重作为热门度
    movie_score = {}
    for m in candidates:
        movie_score[m] = sum(d['weight'] for _, _, d in G.edges(m, data=True))

    result = sorted(movie_score.items(), key=lambda x: x[1], reverse=True)
    return result[:K]


def merge_recommend(rwr_list, comm_list, lam=0.7, K=10):
    score = {}

    for item, s in rwr_list:
        score[item] = score.get(item, 0) + lam * s

    for item, s in comm_list:
        score[item] = score.get(item, 0) + (1 - lam) * s

    result = sorted(score.items(), key=lambda x: x[1], reverse=True)
    return result[:K]


user = 10

rwr_rec = recommend_rwr(G, W, idx, nodes, user, K=10)
comm_rec = recommend_community(G, partition, user, K=10)
final_rec = merge_recommend(rwr_rec, comm_rec, lam=0.7)

print("RWR 推荐：", rwr_rec)
print("社区 推荐：", comm_rec)
print("融合 推荐：", final_rec)
