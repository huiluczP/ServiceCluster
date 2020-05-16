import model.ldaSimpleAdapter as s_lda
import cluster.kmeans as kmn
import data.data_util as du
import os
'''
手动实现lda+KMeans
'''


def s_lda_KMeans(k, file_name, iterate=20):
    # 训练lda模型
    doc_file = du.getDocAsWordArray(file_name)
    lda_model = s_lda.LdaSimpleAdapter(k, doc_file, iterate)
    lda_model.buildModel()
    t_d_t_m = lda_model.print_document_topic_matrix()

    # 聚类
    estimator = kmn.kMeansByFeature(k, t_d_t_m)
    result = s_lda.get_cluster_type_by_doc_topic_matrix(t_d_t_m)
    return t_d_t_m, list(estimator.labels_), result


if __name__ == "__main__":
    t_file_name = "former_text5.csv"
    t_k = 10
    t_iterate = 100

    d_t_m, t_kmn_result, t_result = s_lda_KMeans(t_k, t_file_name, t_iterate)

    print("-----lda cluster result")
    print(t_result)
    print("-----kmn result")
    print(t_kmn_result)