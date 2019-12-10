import matplotlib.pyplot as plt
import random
import numpy as np

color = ["om", "og", "or", "oy", "ok", "ob", "oc", "*m", "*g", "*r"]
text_color = ["purple", "green", "red", "yellow", "black", "blue"]
line_color = ["m-", "g-", "r-", "y-", "k-", "b-", "c-"]


def paintClusterResult(data, label, under_label):
    # 输出对应的纯度，RI，F1，熵的聚类准确度结果矢量图
    # 预计为条形图,data为[[]], 存放对应数据
    x = range(len(data[0]))
    x = [i * 2 for i in x]
    width = 0.26  # 方体宽度
    for i in range(len(data)):
        d = data[i]
        x_place = [p + width * i for p in x]  # 每个方体的x坐标
        rect = plt.bar(x=x_place, height=d, width=width, alpha=0.6, color=text_color[i % len(text_color)], label=label[i])
        for r in rect:
            height = r.get_height()
            plt.text(r.get_x() + r.get_width() / 2, height+0.1, str(round(height, 2)), ha="center", va="bottom")
    plt.ylabel("Num")
    plt.ylim(0, 4)
    plt.xticks([i + width / 2 * (len(data) - 1) for i in x], under_label)
    plt.xlabel("Method")
    plt.title("Clustering Validation Method")
    plt.legend()
    plt.show()


def paintLineChart(method_label, title, result, under_label):
    """
    绘制某个数值不同的方法的折线图
    result计划为对应的不同情况下准确值list，一种method一个list
    method_label为对应方法名，under_label为x轴提示
    """
    x = range(len(under_label))
    x = [i * 3 for i in x]
    t = 0
    for r in result:
        plt.plot(x, r, line_color[t % len(line_color)], linewidth=1, label=method_label[t])
        for i in range(len(r)):
            # 绘制点，更显眼
            plt.plot(x[i], r[i], color[t % len(color)], ms=5)
        t += 1

    plt.title(title)
    plt.ylabel("Num")
    plt.ylim(0, 4)
    plt.xticks(x, under_label)

    plt.legend()
    plt.show()


def printClusterByPointInD2(data, label, result, title, num=5):
    """
    主要将降维处理后的数据进行可视化
    label为对应点的名称，看情况输出
    result为聚类结果，title为图标题
    二维,num为每个类数量
    """
    t = 0
    f_size = 5
    n_count = [0] * len(set(result))  # 计算每个类显示数量
    for p in data:
        if n_count[result[t]] >= num:
            t += 1
            continue
        else:
            font = {
                'weight': 'normal',
                'color': text_color[result[t] % len(text_color)],
                'size': f_size + 2
            }
            n_count[result[t]] += 1
            plt.plot(p[0] * f_size, p[1] * f_size, color[result[t] % len(color)], ms=5)
            plt.text(p[0] * f_size, p[1] * f_size, label[t], fontdict=font)
            t += 1
    plt.title(title)
    plt.show()


def printTopicWordByPointInD2(word_vec, word, result, title, num=5):
    """
    将同一类的topic——word的词向量降维后可视化
    不同类点颜色更改
    每一类最多num个词
    同时输出word_text
    """
    t = 0
    f_size = 5
    n_count = [0] * len(set(result))  # 计算每个类显示数量
    for vec in word_vec:
        if n_count[result[t]] >= num:
            t += 1
            continue
        else:
            font = {
                'weight': 'normal',
                'color': text_color[result[t] % len(text_color)],
                'size': f_size
            }
            n_count[result[t]] += 1
            plt.plot(vec[0], vec[1], color[result[t] % len(color)], ms=5)
            plt.text(vec[0], vec[1], word[t], fontdict=font)
            t += 1

    plt.title(title)
    plt.show()


if __name__ == "__main__":
    """
    # 消去
    data = [[0.23994903737259346, 0.20985124728235321, 2.981690695114797, 0.21973661654578325, 0.2044199643764294],
            [0.22819932049830124, 0.1939356742191813, 3.0694510629684637, 0.18751738275918295, 0.2002154797949506],
            [0.2390996602491506, 0.20107335542436355, 2.9855885411239873, 0.19936212207299242, 0.20290917016588028],
            [0.24561155152887879, 0.22270003964516102, 2.888713427722007, 0.227601311015634, 0.21809874405956886],
            [0.2706681766704417, 0.22103298666103555, 2.8474185711023616, 0.22297060175639938, 0.2187619397297838],
            [0.5676670441676104, 0.5540852363839225, 1.7759887322889758, 0.5618384273973115, 0.5467856379669848],
            [0.6974317817014446, 0.6360511926806193, 1.2573624482141157, 0.6357523547080285, 0.6370693463502655]]

    # 没有消去低频词
    data1 = [[0.20413363533408835, 0.1741365853027712, 3.1142333242952143, 0.17415741491498485, 0.17418452562572398],
             [0.17851075877689693, 0.15905616040804002, 3.1981166712251548, 0.15978134257184162, 0.1578685817878315],
             [0.20682332955832392, 0.18157826689065293, 3.097900007181833, 0.1801289339491933, 0.1828935489465024],
             [0.2357021517553794, 0.22093671940096388, 2.933269496598653, 0.22349519010756314, 0.2184820758677684],
             [0.26174971687429216, 0.23622351499846553, 2.8779525856055117, 0.24230712752956712, 0.23025606944025567],
             [0.49731030577576446, 0.4648843158405803, 2.0249143132505187, 0.44301788753251303, 0.48915237209845586]]

    # text5原始,去除低频词
    data2 = [[0.29554574638844305, 0.24722746259903514, 2.7690420580147816, 0.23947331562993412, 0.2554751179679782],
             [0.2770866773675763, 0.22147899387150877, 2.907332134958868, 0.2061921240124389, 0.23992449429066062],
             [0.2963483146067416, 0.23901167764358275, 2.7968004008055183, 0.2280196732500544, 0.25265656518743795],
             [0.28571428571428575, 0.24850485484466941, 2.7092763940592124, 0.24119648612031452, 0.2561644684650151],
             [0.29534510433386835, 0.27212928479627524, 2.763430125768828, 0.2670682720986832, 0.27792657922830494],
             [0.6386436597110755, 0.6317992246509438, 1.443857180453669, 0.6233095475746683, 0.6411575781641122],
             [0.6974317817014446, 0.6360511926806193, 1.2573624482141157, 0.6357523547080285, 0.6370693463502655]]

    label = ["lda", "tf_idf_keyword_lda", "tf_idf_expand_lda", "w2v_text8", "w2v_api", "btm"]
    under_label = ["purity", "precision", "entropy", "recall", "F1"]
    # paintClusterResult(data2, label, under_label)

    data_line_purity = []
    for i in range(len(label)):
        line_s_list = [data1[i][2], data[i][2]]
        data_line_purity.append(line_s_list)

    under_line = ["fre_0", "fre_5"]
    title_line = "frequent word limit for entropy"
    paintLineChart(label, title_line, data_line_purity, under_line)
    """

    data = [[1.0, 1.0], [0.50287264585495, 0.0], [0.0, 0.9961761236190796]]
    label = ["type1", "type2", "type3"]
    result = [0, 1, 1]
    title = "test"
    printClusterByPointInD2(data, label, result, title)