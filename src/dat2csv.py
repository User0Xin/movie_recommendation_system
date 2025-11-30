# 将dat格式数据转为csv格式数据
import pandas as pd

# 读取dat文件，假设分隔符为'::'
input_file = '../datas/ml-1m/ml-1m/ratings.dat'
output_file = '../datas/ml-1m/ml-1m/ratings.csv'
df = pd.read_csv(input_file, sep='::', engine='python', header=None)
# 为DataFrame添加列名
df.columns = ['userId', 'movieId', 'rating', 'timestamp']
# 保存为csv文件
df.to_csv(output_file, index=False)

input_file = '../datas/ml-1m/ml-1m/movies.dat'
output_file = '../datas/ml-1m/ml-1m/movies.csv'
df = pd.read_csv(input_file, sep='::', engine='python', header=None,encoding='latin-1')
# 为DataFrame添加列名
df.columns = ['movieId', 'title', 'genres']
# 保存为csv文件
df.to_csv(output_file, index=False)