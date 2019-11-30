import model.TF_IDFAdapter as tfidf
import model.ldaAdapter as ldaa
import numpy as np
import cluster.kmeans as kmn

"""
tf_idf预处理，留下keyword
lda模型计算
kMeans聚类
"""

k1 = 3
k2 = 3


def clusterResult(doc={}, num=5):
    # 此时doc为原文本(经过预处理分词)
    # tf_idf处理
    # num 为 keyword数量
    new_doc = tfidf.keyword_by_tf_idf(doc, num)

    # 返回对应的聚类结果
    metrix = []
    ldaa.doc = new_doc
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