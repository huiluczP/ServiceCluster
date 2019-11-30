"""
单独执行btm模型训练的脚本
"""
import model.btmAdapter as btm
import data.data_util as du

if __name__ == "__main__":
    file_name = "test.csv"
    save = "btm_result.txt"
    topic_num = 5  # btm主题数

    document = du.getDocAsWordArray(file_name)
    doc_for_test = {}

    for name_simple in document:
        doc_for_test[name_simple] = document[name_simple]

    model = btm.BtmModel(topic_num, doc_for_test)
    model.buildModel()

    print("输出主题——词")
    model.printTopic_word(5)
    print()

    print("输出文档——主题")
    doc_dis = model.getDoc_Topic()
    for i in range(len(doc_dis)):
        print("文档{} ：{}".format(i, model.name[i]))
        for j in range(len(doc_dis[i])):
            print("\ttopic{}:{}".format(j, doc_dis[i][j]))

    # 分布结果写入文件
    btm.writeResult(doc_dis, save)