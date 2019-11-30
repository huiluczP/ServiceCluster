# 训练聚类测试
import visible.coordinatepainting as cp
import judge.clustereffect as ce
import data.data_util as du

import cluster.lda_kmeans as ldakmn
import cluster.tf_idf_lda_kmeans as lda_ti
import cluster.tf_idf_lda_expand_kmeans as lda_ex_ti
import cluster.tf_idf_w2v_kmeans as w2v_ti
import cluster.btm_kmeans as btm_kmn


def printResult(k, result, former):
    pur = ce.purityClusterResult(k, result, former)
    ri = ce.R1ClusterResult(k, result, former)
    f1 = ce.f1measureClusterResult(k, result, former)
    en = ce.entropyClusterResult(k, result, former)
    pre = ce.precision_cluster(k, result, former)
    recall = ce.recall_cluster(k, result, former)

    print("纯度:{}, RI:{}, F1_measure:{}, 熵：{}, 准确率：{}， 召回率：{}".format(pur, ri, f1, en, pre, recall))
    result_list = [pur, ri, f1, en, pre, recall]
    return result_list


def ldaCluster(k1, k2, filename):
    """
    包含三个参数
    k1为lda模型主题数，k2为聚类数，filename为resource中csv文件
    返回为两个值，第一个为聚类结果，第二个为聚类效果结果
    """
    print("lda模型训练，kMeans聚类")

    # 获取数据
    print("开始获取数据")
    doc = du.getDocAsWordArray(filename)
    # 获取标签信息
    former = du.getFormerCategory(filename)

    # k1主题分类数 k2聚类数量
    ldakmn.k1 = k1
    ldakmn.k2 = k2
    print("lda主题数:{}, 聚类个数:{}".format(ldakmn.k1, ldakmn.k2))
    result = ldakmn.lda_kmn_result(doc)

    result_list = printResult(ldakmn.k2, result, former)
    return result, result_list


def tf_idf_ldaCluster(k1, k2, filename, num):
    """
    包含四个参数
    k1为lda模型主题数，k2为聚类数，filename为resource中csv文件, num为关键词数量
    返回为两个值，第一个为聚类结果，第二个为聚类效果结果
    """
    print("tf_idf预处理, lda模型训练，kMeans聚类")

    # 获取数据
    print("开始获取数据")
    doc = du.getDocAsWordArray(filename)
    # 获取标签信息
    former = du.getFormerCategory(filename)

    # k1主题分类数 k2聚类数量
    lda_ti.k1 = k1
    lda_ti.k2 = k2

    print("lda主题数:{}, 聚类个数:{}".format(ldakmn.k1, ldakmn.k2))
    result = lda_ti.clusterResult(doc, num)

    result_list = printResult(lda_ti.k2, result, former)
    return result, result_list


def tf_idf_expand_lda(k1, k2, filename, num, sim_num):
    """
    包含五个参数
    k1为lda模型主题数，k2为聚类数，filename为resource中csv文件，num为keyword数，sim_num为扩容数
    返回为两个值，第一个为聚类结果，第二个为聚类效果结果
    """
    # 利用tf_idf进行扩容后lda训练
    print("tf_idf扩容预处理, lda模型训练，kMeans聚类")

    # 获取数据
    print("开始获取数据")
    doc = du.getDocAsWordArray(filename)
    # 获取标签信息
    former = du.getFormerCategory(filename)

    # k1主题分类数 k2聚类数量
    lda_ex_ti.k1 = k1
    lda_ex_ti.k2 = k2
    print("lda主题数:{}, 聚类个数:{}".format(ldakmn.k1, ldakmn.k2))
    result = lda_ex_ti.clusterResult(doc, num, sim_num)

    result_list = printResult(lda_ex_ti.k2, result, former)
    return result, result_list


def tf_idf_w2vCluster(k, filename, use_file=False, save_path=r"E:\学校\快乐推荐\word2vec\saveVec"):
    """
    包含四个参数
    k为聚类数，filename为resource中csv文件，use_file为是否使用存储文件
    save_path为W2V的训练文件
    返回为两个值，第一个为聚类结果，第二个为聚类效果结果
    """
    print("tf_idf预处理, w2v训练，kMeans聚类")

    # 获取数据
    doc = du.getDocAsWordArray(filename)
    # 获取标签信息
    former = du.getFormerCategory(filename)

    # 计算，聚类
    if use_file is False:
        result = w2v_ti.clusterResult(doc, k, save_path, False)
    else:
        result = w2v_ti.clusterResultByFile(save_path)

    result_list = printResult(k, result, former)
    return result, result_list


def btmCluster(k, filename, save_file):
    """
    包含二个参数
    k为聚类数，filename为resource中csv文件
    btm比较特殊，model构建较慢，提前进行model创建并使用文件进行读取
    """
    print("BTM模型，KMeans聚类")

    # 获取数据
    doc = du.getDocAsWordArray(filename)
    # 获取标签信息
    former = du.getFormerCategory(filename)

    # 直接计算太慢，直接将处理好的model文件拿来用
    result = btm_kmn.clusterResult(k, save_file)
    result_list = printResult(k, result, former)
    return result, result_list


if __name__ == "__main__":
    topic = 5
    kkt = 3
    file_name = "test.csv"
    label = ["lda", "tf_idf_keyword_lda", "tf_idf_expand_lda", "w2v_text8", "w2v_api", "btm"]

    model_file = "btm_result.txt"  # btm结果文件

    cluster_result1, result1 = ldaCluster(topic, kkt, file_name)
    cluster_result2, result2 = tf_idf_ldaCluster(topic, kkt, file_name, 5)
    cluster_result3, result3 = tf_idf_expand_lda(topic, kkt, file_name, 5, 3)
    cluster_result4, result4 = tf_idf_w2vCluster(kkt, file_name, False, save_path=r"E:\学校\快乐推荐\word2vec\saveVec")
    cluster_result5, result5 = tf_idf_w2vCluster(kkt, file_name, False, save_path=r"E:\学校\快乐推荐\word2vec\api_saveVec")
    cluster_result6, result6 = btmCluster(kkt, file_name, model_file)

    accuracy_result = [result1, result2, result3, result4, result5, result6]
    cp.paintClusterResult(accuracy_result, label)