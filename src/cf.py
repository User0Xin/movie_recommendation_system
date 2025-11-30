import pickle


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
