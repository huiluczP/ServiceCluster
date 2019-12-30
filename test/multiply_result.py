"""
文档主题模型存在随机性
多次计算取准确度的平均值
暂时定为5次重复
"""
import numpy as np
import cluster.cluster_result as res
import visible.coordinatepainting as cp

k1 = 10
k2 = 10
filename = "former_text5.csv"
iterate = 10
key_num = 5
sim_num = 3


def average_result(result_list):
    # 结果集，去掉最高和最低求平均值
    result_type = []
    r = []
    # 将结果集按类型放入矩阵
    if len(result_list) > 0:
        for i in range(len(result_list[0])):
            simple_list = []
            for rest in result_list:
                simple_list.append(rest[i])
            result_type.append(simple_list)
    # 计算平均值
    for tp in result_type:
        tp_min = min(tp)
        tp_max = max(tp)
        tp_sum = sum(tp) - tp_min - tp_max
        r.append(tp_sum)
    return r


def multiple_lda():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.ldaCluster(k1, k2, filename, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_lda_gibbs():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.lda_gibbs_cluster(k1, k2, filename, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_lda_key_word_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_ldaCluster(k1, k2, filename, key_num, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_lda_gibbs_key_word_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_lda_gibbs_Cluster(k1, k2, filename, key_num, sim_num, iterator=500)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_lda_expend_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_expand_lda(k1, k2, filename, key_num, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_lda_gibbs_expend_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_expand_lda_gibbs(k1, k2, filename, key_num, sim_num, iterator=500)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_w2v_tf_idf(save_path):
    whole_result = []
    for i in range(iterate):
        print("迭代：", i)
        cluster_result, result = res.tf_idf_w2vCluster(k2, filename, False, save_path, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_btm(model_file):
    whole_result = []
    for i in range(iterate):
        print("迭代：", i)
        cluster_result, result = res.btmCluster(k2, filename, model_file, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


def multiple_gpu_dmm(model_file):
    whole_result = []
    for i in range(iterate):
        print("迭代：", i)
        cluster_result, result = res.gpu_dmmCluster(k2, filename, model_file, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))
    return cluster_result, list(np.array(whole_result) / it)


if __name__ == "__main__":
    # multiple_lda_gibbs()
    # multiple_lda_gibbs_key_word_ti()
    # multiple_lda_gibbs_expend_ti()
    # multiple_w2v_tf_idf(save_path=r"E:\学校\快乐推荐\word2vec\saveVec")
    multiple_btm(model_file="btm_result_text5_origin.txt")
    # multiple_gpu_dmm(model_file="gpudmm_pdz.txt")
    pass