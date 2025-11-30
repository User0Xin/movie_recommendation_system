import os
import pickle
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split

from models.cf import get_item_cf


def get_all_predictions(trainset):
    predictions = defaultdict(dict)
    for inner_uid in trainset.all_users():
        raw_uid = trainset.to_raw_uid(inner_uid)

        # 用户已评分的电影
        rated_items = set(
            trainset.to_raw_iid(inner_iid) for (inner_iid, rating) in trainset.ur[inner_uid]
        )

        # 遍历所有电影
        for inner_iid in trainset.all_items():
            raw_iid = trainset.to_raw_iid(inner_iid)

            if raw_iid not in rated_items:
                preds = []
                cf_pred = get_item_cf().predict(raw_uid, raw_iid).est
                preds.append(cf_pred)
                # 得分融合，将多个模型的预测结果取平均
                predictions[raw_uid][raw_iid] = np.mean(preds)
    return predictions


def get_all_user_top_k(predictions, K):
    """
    predictions: dict[user][item] = predicted rating
    返回：top_k[user] = [item1, item2, ...]
    """
    top_k = {}
    for uid, user_ratings in predictions.items():
        # sort by predicted score
        ranked_items = sorted(user_ratings.items(), key=lambda x: x[1], reverse=True)
        top_k[uid] = [iid for iid, score in ranked_items[:K]]
    return top_k


def build_fuse_predict():
    load_dotenv()
    # 加载数据
    file_path = os.path.join(os.getenv('FILE_PATH'), "ratings.dat")
    # Movielens 1M format: UserID::MovieID::Rating::Timestamp
    reader = Reader(
        line_format='user item rating timestamp',
        sep='::',
        rating_scale=(1, 5)
    )
    data = Dataset.load_from_file(file_path, reader=reader)
    trainset, _ = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    print("Building fused prediction model...")
    # 获取所有用户-物品预测评分
    predictions = get_all_predictions(trainset)
    print("Fused prediction model built.")
    # 获取用户Top-20推荐列表
    fuse_top20 = get_all_user_top_k(predictions, 20)
    # 持久化保存
    with open('../predicts/fuse_pred', 'wb') as f:
        pickle.dump(predictions, f)
    with open('../predicts/fuse_top20', 'wb') as f:
        pickle.dump(fuse_top20, f)


def fuse_recommend(uid, k):
    with open('../predicts/fuse_top20', 'rb') as f:
        fuse_top20 = pickle.load(f)
    top_k = fuse_top20[uid][:k]
    return top_k

if __name__ == '__main__':
    build_fuse_predict()
    print(fuse_recommend('1', 10))
