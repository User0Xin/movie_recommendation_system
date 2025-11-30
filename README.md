# 电影推荐系统
## datas
存放数据集的文件夹。
## models
存放模型文件夹。
模型需提供一个`predict(user_id, movie_id)`方法，用于根据用户ID和电影ID进行评分预测。
## predicts
存放预测结果的文件夹。用于离线计算预测结果。避免线上实时计算，提高响应速度。
## src
存放源码的文件夹。
主要包含以下模块：
 - build_cf_predicts.py: 构建协同过滤预测结果的脚本。
 - build_graph.py: 将数据存入Neo4j的脚本，目前暂未用到Neo4j数据后续可能使用RAG对接。
 - evaluate.py: 评估模型性能的脚本。
 - fuse_predicts.py: 融合多个模型预测结果的脚本，调用每个模型的predict方法通过线性融合得到综合评分。
 - utils.py: 工具函数模块。