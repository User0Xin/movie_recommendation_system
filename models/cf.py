import os
import pickle

from dotenv import load_dotenv
from surprise import Reader, Dataset, KNNBaseline
from surprise.model_selection import train_test_split

from src.config import FILE_PATH

item_cf = None


def get_item_cf():
    global item_cf
    if item_cf is not None:
        return item_cf

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
    trainset, testset = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)
    sim_options_item = {
        "name": "cosine",
        "user_based": False
    }
    item_cf = KNNBaseline(sim_options=sim_options_item)
    item_cf.fit(trainset)

    return item_cf


def get_user_top_k_by_userCF(uid, k):
    with open('../predicts/user_top20', 'rb') as f:  # 注意是二进制读取模式
        user_top20 = pickle.load(f)
    top_k = user_top20[uid][:k]
    return top_k


def get_user_top_k_by_itemCF(uid, k):
    with open('../predicts/item_top20', 'rb') as f:  # 注意是二进制读取模式
        item_top20 = pickle.load(f)
    top_k = item_top20[uid][:k]
    return top_k


if __name__ == '__main__':
    userId = '1'
    K = 10
    top_k = get_user_top_k_by_userCF(userId, K)
    print(f"Top {K} recommendations for user {userId}: {top_k}")
    top_k_item = get_user_top_k_by_itemCF(userId, K)
    print(f"Top {K} recommendations for user {userId} by ItemCF: {top_k_item}")
