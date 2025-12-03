import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM
import pickle # 用于保存 LabelEncoder
import torch
import os
# 全局变量或配置，确保加载时一致
EMBEDDING_DIM = 16
model_path='../DeepFM.pth'
lbe_path='../DeepFM_lbe_encoders.pkl'



def save_system(model, lbe_dict, model_path, lbe_path):
    """保存模型参数和编码器"""
    print(f"正在保存模型到 {model_path} ...")
    torch.save(model.state_dict(), model_path)

    print(f"正在保存编码器到 {lbe_path} ...")
    with open(lbe_path, 'wb') as f:
        pickle.dump(lbe_dict, f)
    print("保存完成。")


def load_system(model_path='DeepFM.pth', lbe_path='DeepFM_lbe_encoders.pkl', sparse_features=None):
    """
    加载系统：
    1. 读取编码器 (为了知道每个特征有多少个取值 vocabulary_size)
    2. 重新构建模型结构
    3. 加载模型参数
    """
    if not os.path.exists(model_path) or not os.path.exists(lbe_path):
        raise FileNotFoundError("找不到模型文件或编码器文件！")

    # 1. 加载编码器
    print("正在加载编码器...")
    with open(lbe_path, 'rb') as f:
        lbe_dict = pickle.load(f)

    # 2. 重新定义 Feature Columns (这一步必须和训练时完全一致)
    # 我们需要利用加载回来的 lbe_dict 来获取 vocabulary_size
    from deepctr_torch.inputs import SparseFeat
    from deepctr_torch.models import DeepFM

    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=len(lbe_dict[feat].classes_), embedding_dim=EMBEDDING_DIM)
        for feat in sparse_features
    ]
    linear_cols = fixlen_feature_columns
    dnn_cols = fixlen_feature_columns

    # 3. 初始化模型结构
    model = DeepFM(linear_cols, dnn_cols, task='regression', device='cpu')

    # 4. 加载参数权重
    print("正在加载模型参数...")
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # 切换到评估模式

    return model, lbe_dict

def load_and_merge_data(data_dir='../datas/ml-1m/ml-1m'):
    """
    1. 读取 ratings, users, movies 三个文件
    2. 将它们合并成一个大的 DataFrame
    """
    print(f"Loading data from {data_dir}...")

    # 读取 Ratings
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv(f'{data_dir}/ratings.dat', sep='::', header=None, names=r_cols, engine='python')

    # 读取 Users
    u_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    users = pd.read_csv(f'{data_dir}/users.dat', sep='::', header=None, names=u_cols, engine='python')

    # 读取 Movies
    m_cols = ['movie_id', 'title', 'genres']
    movies = pd.read_csv(f'{data_dir}/movies.dat', sep='::', header=None, names=m_cols, engine='python',
                         encoding='ISO-8859-1')

    # 合并数据
    # 先合并 ratings 和 users，再合并 movies
    data = pd.merge(pd.merge(ratings, users, on='user_id'), movies, on='movie_id')

    print(f"Data loaded successfully. Total samples: {len(data)}")
    return data


def preprocess_data(data, sparse_features, target_col='rating', test_size=0.2, embedding_dim=16):
    """
    1. 对稀疏特征进行 Label Encoding
    2. 生成 DeepCTR 需要的 Feature Columns 配置
    3. 划分训练集和测试集
    4. 生成模型输入的字典格式
    """
    print("Preprocessing data (Label Encoding)...")

    #训练前收集 lbe_dict，并在训练后调用保存
    lbe_dict = {}

    # 1. Label Encoding
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        lbe_dict[feat] = lbe

    # 2. 定义 Feature Columns
    # 计算每个特征有多少个取值 (vocabulary size)
    fixlen_feature_columns = [
        SparseFeat(feat, vocabulary_size=data[feat].max() + 1, embedding_dim=embedding_dim)
        for feat in sparse_features
    ]

    # DeepFM 的线性部分(Linear)和深度部分(DNN)通常共享输入特征
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # 3. 划分数据集
    train, test = train_test_split(data, test_size=test_size, random_state=2024)

    # 4. 生成字典格式输入 (Model Input)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    return train_model_input, test_model_input, train[target_col].values, test[
        target_col].values, linear_feature_columns, dnn_feature_columns, lbe_dict


def train_model(train_input, train_target, linear_cols, dnn_cols, batch_size=256, epochs=3):
    """
    1. 初始化 DeepFM 模型
    2. 编译并训练
    """
    print("Initializing and training DeepFM...")

    # 初始化模型
    # task='regression' 用于评分预测, device='auto' 会自动检测 GPU
    model = DeepFM(linear_cols, dnn_cols, task='regression', device='cuda')

    # 编译模型
    model.compile("adam", "mse", metrics=['mse'])

    # 训练
    history = model.fit(train_input, train_target, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_split=0.1)

    return model, history


def evaluate_model(model, test_input, test_target):
    """
    1. 在测试集上预测
    2. 计算 RMSE
    """
    print("Evaluating model...")

    pred_ans = model.predict(test_input, batch_size=256)

    # 计算 MSE 和 RMSE
    mse = ((test_target - pred_ans.flatten()) ** 2).mean()
    rmse = mse ** 0.5

    print(f"Test RMSE: {rmse:.4f}")
    return rmse


def get_user_feature_dict(data_df, user_id):
    """辅助函数：从数据集中提取指定用户的特征（年龄、性别等）"""
    # 找到该用户的第一条记录（因为用户特征在所有行都是一样的）
    user_row = data_df[data_df['user_id'] == user_id].iloc[0]
    return user_row


def get_movie_feature_dict(data_df, movie_id):
    """辅助函数：从数据集中提取指定电影的特征（题材等）"""
    movie_row = data_df[data_df['movie_id'] == movie_id].iloc[0]
    return movie_row


def predict_single_rating(model, data_df, user_id, movie_id, feature_names):
    """
    功能：预测单个用户对单个电影的评分
    """
    # 1. 准备特征数据
    # 注意：这里的 user_id 和 movie_id 必须是经过 LabelEncoding 之后的数值
    # 如果找不到（比如是新用户），这里会报错，实际工程中需要做冷启动处理
    try:
        user_feat = get_user_feature_dict(data_df, user_id)
        movie_feat = get_movie_feature_dict(data_df, movie_id)
    except IndexError:
        print(f"Error: User {user_id} or Movie {movie_id} not found in dataset.")
        return None

    # 2. 组装模型输入 (字典格式)
    model_input = {}
    for name in feature_names:
        # 如果特征名字属于用户特征，取用户的值；否则取电影的值
        if name in ['user_id', 'gender', 'age', 'occupation', 'zip']:
            model_input[name] = np.array([user_feat[name]])
        else:  # movie_id, genres, title
            model_input[name] = np.array([movie_feat[name]])

    # 3. 预测
    pred_score = model.predict(model_input, batch_size=1)
    return pred_score[0][0]


def recommend_top_n(model, data_df, user_id, feature_names, top_n=5):
    """
    功能：为用户推荐预测评分最高的 N 部电影
    逻辑：把所有电影拉出来，给这个用户都预测一遍，然后排序
    """
    print(f"正在为用户 {user_id} 生成推荐列表...")

    # 1. 获取该用户的特征
    user_feat = get_user_feature_dict(data_df, user_id)

    # 2. 获取所有唯一的电影 ID 及其对应的特征
    # 为了去重，我们按 movie_id 分组取第一条即可
    unique_movies = data_df.drop_duplicates(subset=['movie_id'])

    # 3. 构造批量预测的输入数据
    num_movies = len(unique_movies)
    model_input = {}

    for name in feature_names:
        if name in ['user_id', 'gender', 'age', 'occupation', 'zip']:
            # 用户特征复制 N 份 (N = 电影总数)
            model_input[name] = np.full(num_movies, user_feat[name])
        else:
            # 电影特征直接使用列表
            model_input[name] = unique_movies[name].values

    # 4. 批量预测
    preds = model.predict(model_input, batch_size=256)

    # 5. 整理结果
    # 将预测分数和电影ID、电影标题拼在一起
    results = pd.DataFrame({
        'movie_id': unique_movies['movie_id'],
        'title': unique_movies['title'],  # 假设之前没有 drop 掉 title
        'pred_score': preds.flatten()
    })

    # 6. 排序并取 Top N
    top_movies = results.sort_values(by='pred_score', ascending=False).head(top_n)

    return top_movies

def predict(raw_uid = 1, raw_mid = 1193):
    #加载数据
    sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "genres"]
    data_df = load_and_merge_data(data_dir='../datas/ml-1m/ml-1m')
    _, _, _, _, linear_cols, dnn_cols, _ = preprocess_data(
        data_df,
        sparse_features,
        target_col='rating'
    )
    feature_names = get_feature_names(linear_cols + dnn_cols)
    # 1. 加载模型
    loaded_model, loaded_lbe = load_system(
        model_path,
        lbe_path,
        sparse_features=sparse_features
    )
    # 2. 预测一个新样本
    # 比如预测 UserID (原始ID)=1 对 MovieID (原始ID)=1193 的评分
    # 注意：必须用加载回来的编码器将 原始ID 转为 内部ID
    # transform 接受列表，所以加 []，最后取 [0]
    target_uid = loaded_lbe['user_id'].transform([raw_uid])[0]
    target_mid = loaded_lbe['movie_id'].transform([raw_mid])[0]

    score = predict_single_rating(loaded_model, data_df, target_uid, target_mid, feature_names)
    print(f"\n预测用户 {target_uid} 对电影 {target_mid} 的评分: {score:.4f}")

    # ----------------------------------------------------
    # 测试 2: 给用户 10 推荐 5 部电影
    # ----------------------------------------------------
    # recommendations = recommend_top_n(loaded_model, data_df, user_id=target_uid, feature_names=feature_names, top_n=5)
    # 
    # print(f"\n用户 {target_uid} 的 Top 5 推荐电影:")
    # print(recommendations[['movie_id', 'pred_score']])  # 这里 title 可能被编码了，看不出名字，主要看 ID
    return score

if __name__ == "__main__":
    # 1. 定义特征配置
    # 这里我们定义哪些列是类别型特征
    sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "genres"]

    # 2. 加载数据
    data_df = load_and_merge_data(data_dir='../datas/ml-1m/ml-1m')

    # 3. 数据预处理
    train_in, test_in, train_y, test_y, linear_cols, dnn_cols, lbe_dict = preprocess_data(
        data_df,
        sparse_features,
        target_col='rating'
    )

    # 4. 训练模型
    model, history = train_model(train_in, train_y, linear_cols, dnn_cols, epochs=3)

    # 5. 评估
    evaluate_model(model, test_in, test_y)

    # 6. 保存模型
    save_system(model, lbe_dict)

    # 7. 预测
    # 获取特征名列表 (DeepCTR 需要这个来构建字典)
    feature_names = get_feature_names(linear_cols + dnn_cols)

    # ----------------------------------------------------
    # 测试 1: 预测特定的一对 (假设 UserID=10, MovieID=20)
    # ----------------------------------------------------
    # 注意：这里的 ID 是编码后的内部 ID (0~6039)，不是原始的 CSV 里的 ID
    target_uid = 10
    target_mid = 20

    score = predict_single_rating(model, data_df, target_uid, target_mid, feature_names)
    print(f"\n预测用户 {target_uid} 对电影 {target_mid} 的评分: {score:.4f}")

    # ----------------------------------------------------
    # 测试 2: 给用户 10 推荐 5 部电影
    # ----------------------------------------------------
    recommendations = recommend_top_n(model, data_df, user_id=target_uid, feature_names=feature_names, top_n=5)

    print(f"\n用户 {target_uid} 的 Top 5 推荐电影:")
    print(recommendations[['movie_id', 'pred_score']])  # 这里 title 可能被编码了，看不出名字，主要看 ID
