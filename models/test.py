from my_predict import predict

u = 4169
m = 1898
score = predict(u, m)  # 第一次会加载整张表，之后都是内存查表
print(score)
