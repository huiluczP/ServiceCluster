import model.btmAdapter as btm
import numpy as np
import cluster.kmeans as kmn
import os
"""
利用btm进行模型训练，获取分布信息
利用kMeans进行聚类处理
后续可能会进行相似度预处理
"""


def clusterResult(k, file_name):
    # 获取路径信息
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find("ServiceCluster\\") + len("ServiceCluster\\")]
    path = os.path.abspath(rootPath + "resource\\" + file_name)

    # 获取已经创建好的模型信息
    print("加载文档-主题矩阵")
    result = btm.loadModel(path)
    data = np.array(result)

    print("开始kMeans聚类")
    estimator = kmn.kMeansByFeature(k, data)
    print("聚类完成")

    return list(estimator.labels_)


if __name__ == "__main__":
    t_result = clusterResult(3, "btm_result.txt")
    print(t_result)