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
filename = "mashup_test5.csv"
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
        cluster_result, result = res.ldaCluster(k1, k2, filename)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))


def multiple_lda_key_word_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_ldaCluster(k1, k2, filename, key_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))


def multiple_lda_expend_ti():
    whole_result = []
    for i in range(iterate):
        cluster_result, result = res.tf_idf_expand_lda(k1, k2, filename, key_num, sim_num)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))


def multiple_w2v_tf_idf(save_path):
    whole_result = []
    for i in range(iterate):
        print("迭代：", i)
        cluster_result, result = res.tf_idf_w2vCluster(k2, filename, False, save_path)
        whole_result.append(result)
    whole_result = average_result(whole_result)
    it = iterate - 2
    print(list(np.array(whole_result) / it))


if __name__ == "__main__":
    # multiple_lda()
    # multiple_lda_key_word_ti()
    # multiple_lda_expend_ti()
    multiple_w2v_tf_idf(save_path=r"E:\学校\快乐推荐\word2vec\saveVec")
    # multiple_w2v_tf_idf(save_path=r"E:\学校\快乐推荐\word2vec\api_saveVec")
    pass