import math
import os
import pickle
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

from src.build_cf_predicts import get_all_user_top_k


def getTrueItem():
    load_dotenv()
    # # 1. Load  dataset
    file_path = os.path.join(os.getenv('FILE_PATH'), "ratings.dat")

    # Movielens 1M format: UserID::MovieID::Rating::Timestamp
    reader = Reader(
        line_format='user item rating timestamp',
        sep='::',
        rating_scale=(1, 5)
    )

    data = Dataset.load_from_file(file_path, reader=reader)

    # train-test split
    trainset, testset = train_test_split(data, test_size=0.2,shuffle=True, random_state=42)

    # 将 testset 转为 dict: true_items[user] = {item1, item2, ...}
    true_items = defaultdict(set)
    for uid, iid, rating in testset:
        if rating >= 3:  # 认为 >=3 的评分为用户喜欢的
            true_items[uid].add(iid)
    return true_items


def ndcg_at_k(recommended_list, true_items):
    dcg = 0.0
    for idx, item in enumerate(recommended_list):
        if item in true_items:
            dcg += 1 / math.log2(idx + 2)  # rank position starts at 1
    # ideal DCG
    idcg = sum([1 / math.log2(i + 2) for i in range(min(len(true_items), len(recommended_list)))])
    return dcg / idcg if idcg > 0 else 0


def evaluate(top_k, true_items, K):
    precision_list = []
    recall_list = []
    hitrate_list = []
    ndcg_list = []

    for uid in top_k:
        recs = top_k[uid]
        trues = true_items.get(uid, set())

        if len(trues) == 0:
            continue

        hits = len(set(recs) & trues)

        precision_list.append(hits / K)
        recall_list.append(hits / len(trues))
        hitrate_list.append(1 if hits > 0 else 0)
        ndcg_list.append(ndcg_at_k(recs, trues))

    return {
        "Precision@K": np.mean(precision_list),
        "Recall@K": np.mean(recall_list),
        "HitRate@K": np.mean(hitrate_list),
        "NDCG@K": np.mean(ndcg_list),
    }


if __name__ == '__main__':
    K = 20
    #
    # with open('../predicts/user_pred', 'rb') as f:  # 注意是二进制读取模式
    #     user_pred = pickle.load(f)
    # with open('../predicts/item_pred', 'rb') as f:  # 注意是二进制读取模式
    #     item_pred = pickle.load(f)
    #
    # topk_user = get_all_user_top_k(user_pred, K)
    # topk_item = get_all_user_top_k(item_pred, K)
    #
    # metrics_user = evaluate(topk_user, getTrueItem(), K)
    # metrics_item = evaluate(topk_item, getTrueItem(), K)

    # print("User-based CF Metrics:", metrics_user)
    # print("Item-based CF Metrics:", metrics_item)

    with open('../predicts/fuse_pred', 'rb') as f:  # 注意是二进制读取模式
        fuse_pred = pickle.load(f)

    topk_user = get_all_user_top_k(fuse_pred, K)
    metrics_user = evaluate(topk_user, getTrueItem(), K)
    print("User-based CF Metrics:", metrics_user)

