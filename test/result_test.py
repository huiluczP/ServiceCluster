import cluster.cluster_result as res
import visible.coordinatepainting as cp

if __name__ == "__main__":
    topic = 5
    kkt = 3
    file_name = "test.csv"
    label = ["lda", "tf_idf_keyword_lda", "tf_idf_expand_lda", "w2v_text8", "w2v_api", "btm"]

    model_file = "btm_result.txt"  # btm结果文件

    cluster_result1, result1 = res.ldaCluster(topic, kkt, file_name)
    cluster_result2, result2 = res.tf_idf_ldaCluster(topic, kkt, file_name, 5)
    cluster_result3, result3 = res.tf_idf_expand_lda(topic, kkt, file_name, 5, 3)
    cluster_result4, result4 = res.tf_idf_w2vCluster(kkt, file_name, False, save_path=r"E:\学校\快乐推荐\word2vec\saveVec")
    cluster_result5, result5 = res.tf_idf_w2vCluster(kkt, file_name, False, save_path=r"E:\学校\快乐推荐\word2vec\api_saveVec")
    cluster_result6, result6 = res.btmCluster(kkt, file_name, model_file)

    accuracy_result = [result1, result2, result3, result4, result5, result6]
    cp.paintClusterResult(accuracy_result, label)