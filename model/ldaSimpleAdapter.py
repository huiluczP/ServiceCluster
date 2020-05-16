import numpy as np
import os
import random
import data.data_util as du

"""
lda手工实现，准确度会低一点
"""


def get_cluster_type_by_doc_topic_matrix(doc_topic_matrix):
    # 公共方法，直接按照文档主题矩阵概率中概率分布最高的分量作为文档的主题归属
    # 形式和KMeans那边类似，[0,0,1,1,1,2,2...]
    topic_result = []
    for distribution in doc_topic_matrix:
        max_value = max(distribution)
        index = list(distribution).index(max_value)
        topic_result.append(index)
    return topic_result


class LdaSimpleAdapter:
    doc = {}
    name = []  # 存对应的文档名，便于之后确认
    id_word_dic = {}  # 词典，id-word对
    word_dic = {}  # 词典，word-id对
    word_list = []  # doc中词列表，每个文档一list
    doc_word_id = []  # doc中的词映射为id
    topic_sum = []  # 每个主题对应词汇数
    topic_word_sum = None  # 存放每个主题中词汇表中词被分入的次数
    word_topic_list = []  # 当前每个doc中每个词汇的topic

    def __init__(self, num, doc, iterate=100):
        # 设置超参数和迭代次数
        self.k = num
        self.alpha = 50 / self.k
        self.beta = 0.1
        self.iterate = iterate  # 迭代次数
        self.doc = doc

    def init_dic_word(self):
        # 处理doc，将编号词汇对和词汇编号对存入字典
        for d in self.doc:
            doc_value = self.doc[d]
            words = str(doc_value).split(" ")
            self.name.append(d)
            self.word_list.append(words)

        # 编号
        index = 0
        for simple_word_list in self.word_list:
            for w in simple_word_list:
                if w not in self.word_dic:
                    # 未被存入的词汇
                    self.word_dic[w] = index
                    self.id_word_dic[index] = w
                    index += 1

    def init_doc_word_to_id(self):
        # 根据完成初始化的词汇-id字典将doc处理成id vec
        for simple_word_list in self.word_list:
            simple_id_list = []
            for w in simple_word_list:
                simple_id = self.word_dic[w]
                simple_id_list.append(simple_id)
            self.doc_word_id.append(simple_id_list)

    def init_matrix(self):
        # 初始化主题分配，随机对每个词分配主题
        v = len(self.word_dic)
        self.topic_sum = [0] * self.k
        self.topic_word_sum = np.zeros((self.k, len(self.word_dic)))  # len(topic) * len(vocabulary)
        for simple_id_list in self.doc_word_id:
            # 随机分配
            simple_word_topic_list = [0] * len(simple_id_list)
            for i in range(len(simple_id_list)):
                topic = random.randint(0, self.k - 1)
                simple_word_topic_list[i] = topic
                self.topic_sum[topic] += 1
                self.topic_word_sum[topic][simple_id_list[i]] += 1
            self.word_topic_list.append(simple_word_topic_list)

    def gibbs(self, d_index, w_index):
        # 吉布斯采样,参数为文档序号和对应词序号
        topic = self.word_topic_list[d_index][w_index]
        # 该词以外条件
        self.topic_sum[topic] -= 1
        word_id = self.doc_word_id[d_index][w_index]
        self.topic_word_sum[topic][word_id] -= 1
        # 采样
        topic = self.sampling(topic, d_index, word_id)
        self.word_topic_list[d_index][w_index] = topic
        # 恢复该词条件
        self.topic_sum[topic] += 1
        self.topic_word_sum[topic][word_id] += 1

    def sampling(self, topic, d_index, word_id):
        # 采样公式,计算概率分布
        p = [0.0] * self.k
        simple_doc_word_topic = self.word_topic_list[d_index]
        # 对每个主题进行计算
        for i in range(self.k):
            n_d = list(simple_doc_word_topic).count(topic) - 1  # 该文档中除了该词以外的主题k的词汇数量
            n_d_all = len(simple_doc_word_topic) - 1  # 该文档除了该词以外的词汇数目
            n_w = self.topic_word_sum[topic][word_id]  # 该词被分入该主题的次数
            n_w_all = self.topic_sum[topic]  # 词汇被分入该主题的次数
            p[i] = (n_d + self.alpha) / (n_d_all + self.k * self.alpha) * \
                (n_w + self.beta) / (n_w_all + len(self.word_dic) * self.beta)
        # 轮盘赌选择主题
        for j in range(len(p)):
            if j == 0:
                continue
            p[j] = p[j - 1] + p[j]
        ran = random.random()
        for k in range(len(p)):
            if ran * p[self.k - 1] <= p[k]:
                return k
        return 0

    def buildModel(self):
        # 模型训练
        print("初始化文档字典------")
        self.init_dic_word()
        print("初始化词汇表------")
        self.init_doc_word_to_id()
        print("初始化统计信息------")
        self.init_matrix()
        print("开始迭代------")
        for i in range(self.iterate):
            print("迭代次数：{}".format(i))
            for j in range(len(self.word_list)):
                for k in range(len(self.word_list[j])):
                    self.gibbs(j, k)

    def print_document_topic_matrix(self):
        # 计算文档主题矩阵
        doc_topic_matrix = np.zeros((len(self.doc), self.k))
        for i in range(len(self.doc)):
            for j in range(self.k):
                # 等于该文档中当前主题词汇数量+alpha/当前文档词汇数+alpha*k
                dis = (list(self.word_topic_list[i]).count(j) + self.alpha) / (len(self.word_topic_list[i]) + self.alpha * self.k)
                doc_topic_matrix[i][j] = dis

        for distribution in doc_topic_matrix:
            print(distribution)
        return doc_topic_matrix


if __name__ == "__main__":
    file_name = "former_text5.csv"
    doc_file = du.getDocAsWordArray(file_name)
    k_num = 10
    t_iterate = 100

    lda_model = LdaSimpleAdapter(k_num, doc_file, t_iterate)
    lda_model.buildModel()
    t_d_t_m = lda_model.print_document_topic_matrix()
    t_result = get_cluster_type_by_doc_topic_matrix(t_d_t_m)
    print(t_result)