from cluster import kmeans as kmn
from model import ldaAdapter as ldaa
import numpy as np

'''
实现将lda的结果作为聚类特征
同时进行kMeans聚类并绘制图像
'''

k1 = 3
k2 = 3


def lda_kmn_result(doc={}):
    # 返回对应的聚类结果
    metrix = []
    ldaa.doc = doc
    # 获取lda模型和词袋
    print("创建主题模型")
    lda, corpus = ldaa.ldaAdapter(k1)
    for index, values in enumerate(lda.inference(corpus)[0]):
        topicmatrix = []
        # 对应文档的分布
        for topic, value in enumerate(values):
            topicmatrix.append(value)
        metrix.append(topicmatrix)

    # 将metrix送入kMeans进行聚类
    print("开始kMeans聚类")
    data = np.array(metrix)
    estimator = kmn.kMeansByFeature(k2, metrix)

    labels = list(estimator.labels_)
    return labels

