from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

def SVDmodel(file_path, n_factors=100, n_epochs=20):
    """
    训练一个SVD模型。

    Args:
        file_path (str): 数据集文件路径。
        n_factors (int): 奇异值分解的因子数量。
        n_epochs (int): 训练的迭代次数。

    Returns:
        训练好的SVD模型。
    """
    # 1. 加载数据
    reader = Reader(line_format='user item rating timestamp', sep='::')
    data = Dataset.load_from_file(file_path, reader=reader)

    # 2. 划分训练集和测试集
    trainset, testset = train_test_split(data, test_size=0.25)

    # 3. 初始化并训练 SVD 模型
    algo = SVD(n_factors, n_epochs)  # n_factors是隐特征数量
    algo.fit(trainset)
    return algo

def predict(algo, uid = '1', iid = '1193'):
    """
    使用训练好的模型进行预测。

    Args:
        algo: 训练好的SVD模型。
        uid (str): 用户ID。
        iid (str): 物品ID。

    Returns:
        预测结果。
    """
    # 预测一个特定评分
    # 假设我们要预测 UserID='1' 对 MovieID='1193' 的评分
    pred = algo.predict(uid, iid, verbose=True)
    return pred.est
    # 5. 在测试集上评估整体准确率 (RMSE)
    #predictions = algo.test(testset)
    #accuracy.rmse(predictions)

if __name__ == '__main__':
    file_path = '../datas/ml-1m/ml-1m/ratings.dat'
    algo = SVDmodel(file_path, n_factors=100, n_epochs=20)
    pred=predict(algo, uid = '1', iid = '1193')
    print(pred)
