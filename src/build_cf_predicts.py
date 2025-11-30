import os
import pickle

from surprise import Dataset, Reader, KNNBasic, KNNBaseline
from surprise.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import math
from dotenv import load_dotenv

from src.config import FILE_PATH


def get_all_predictions(model, trainset):
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
                pred = model.predict(raw_uid, raw_iid)
                predictions[raw_uid][raw_iid] = pred.est

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


def build_cf_predict():
    load_dotenv()
    # # 1. Load  dataset
    file_path = FILE_PATH / "ratings.dat"
    # Movielens 1M format: UserID::MovieID::Rating::Timestamp
    reader = Reader(
        line_format='user item rating timestamp',
        sep='::',
        rating_scale=(1, 5)
    )
    data = Dataset.load_from_file(file_path, reader=reader)
    # train-test split
    trainset, testset = train_test_split(data, test_size=0.2,shuffle=True, random_state=42)
    # 2. User-based CF
    sim_options_user = {
        "name": "cosine",
        "user_based": True
    }
    user_cf = KNNBaseline(sim_options=sim_options_user)
    user_cf.fit(trainset)
    # 3. Item-based CF
    sim_options_item = {
        "name": "cosine",
        "user_based": False
    }
    item_cf = KNNBaseline(sim_options=sim_options_item)
    item_cf.fit(trainset)
    user_pred = get_all_predictions(user_cf, trainset)
    item_pred = get_all_predictions(item_cf, trainset)
    user_top20 = get_all_user_top_k(user_pred, 20)
    item_top20 = get_all_user_top_k(item_pred, 20)
    with open('../predicts/user_pred', 'wb') as f:  # 注意是二进制写入模式
        pickle.dump(user_pred, f)
    with open('../predicts/item_pred', 'wb') as f:  # 注意是二进制写入模式
        pickle.dump(item_pred, f)
    with open('../predicts/user_top20', 'wb') as f:  # 注意是二进制写入模式
        pickle.dump(user_top20, f)
    with open('../predicts/item_top20', 'wb') as f:  # 注意是二进制写入模式
        pickle.dump(item_top20, f)



if __name__ == '__main__':
    build_cf_predict()
