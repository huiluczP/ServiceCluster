# ServiceCluster
本项目包含五种不同文本模型训练策略，六种聚类准确度判断方法实现

## 文本模型
  * LDA模型
  * TF_IDF预处理，保留关键词，LDA模型  
  * TF_IDF预处理，关键词扩容，LDA模型
  * TF_IDF+Word2Vec, 计算文档矩阵
  * BTM Biterm Topic Model模型

## 聚类方法
    kMeans，cluster.kmeans文件
	
## 聚类准确度评价方法
    纯度，熵，RI，F1，准确率，召回率
    其中RI与F1可能比较奇怪，对于多类和少类比较敏感，不用特别在意
    准确率与召回率参考论文[1]实现

## 文档设置方法
    将csv文件放入resource中，在调用时写入对应文件名即可
	
##  返回格式
    cluster聚类返回格式为每个文档的类别list，例如[0,1,1,2,2]，表示聚类簇数为3
	
##  使用方法
    cluster.cluster_result中调用各个模型聚类方法即可
    返回为双返回，聚类结果与聚类准确度评价结果
	
    在关键词扩容和Word2vec相关方法中，需要输入对应的Word2Vec完成训练的文件才可使用
    训练代码在`word2vecAdapter`文件中实现
	
    BTM因为比较耗时，所以先使用训练代码进行训练并保存训练结果后在将结果文件作为参数进行聚类
    训练流程在`test.build_btm_model`文件中实现，可参考

    `result_test`文件中有整个流程的示例，可参考

##  参考
    [1]  Tian G, He K, Wang J, Sun C, Xu J. Domain-oriented and Tag-aided Web Service Clustering Method. Acta Electronica Sinica, 2015, 43(7), 1266-1274. (田刚, 何克清, 王健, 孙承爱, 徐建建. 面向领域标签辅助的服务聚类方法. 电子学报, 2015, 43(7), 1266-1274.)
