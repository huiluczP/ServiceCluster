import model.TF_IDFAdapter as tfidf
import model.ldaAdapter as ldaa
import model.word2vecAdapter as w2v
import numpy as np
import cluster.kmeans as kmn

"""
tf_idf预处理，w2v扩展文本
lda模型计算
kMeans聚类
"""

k1 = 3  # lda主题数
k2 = 3  # 聚类数


def clusterResult(doc={}, num=5, sim_num=3):
    # 此时doc为原文本(经过预处理分词)
    # 对文本进行处理，获取tf——idf，使用w2v进行扩容
    # 预计取前5个，扩容为3个
    # num 为 keyword数量
    # sim_num 为 扩容数
    model = w2v.load_model_binary(r"E:\学校\快乐推荐\word2vec\api_saveVec")
    doc = tfidf.expend_word(model, doc, num, sim_num)

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